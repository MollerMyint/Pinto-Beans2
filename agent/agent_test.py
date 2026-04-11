# file to create chat agent to help students navigate website

import os
import time
import json
import sqlite3
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from functools import lru_cache
from sentence_transformers import SentenceTransformer

load_dotenv() # load environment variables from .env file

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "campus_kb.db"
CORPUS_DIR = BASE_DIR / "itc2026_ai_corpus"  # raw .md files for brute-force search
# how many likely matches we keep before doing deeper semantic scoring
SEMANTIC_CANDIDATE_LIMIT = 300
MIN_TOKEN_LEN = 2
# common words to ignore so search focuses on meaningful terms
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "what", "when",
    "where", "which", "who", "why", "with", "your", "you", "i", "me", "my", "we",
    "our", "us", "cal", "poly", "pomona"
}


def _tokenize_query(query: str) -> list[str]:
    """
    Normalize query text into alphanumeric tokens for robust FTS matching.
    """
    cleaned = (query or "").strip().lower()
    if not cleaned:
        return []
    tokens = re.findall(r"[a-z0-9]+", cleaned) # find all alphanumeric tokens
    return [t for t in tokens if len(t) >= MIN_TOKEN_LEN] # return tokens with minimum length


def _stopping_tokens(query: str) -> list[str]:
    """
    Remove stopwords from query.
    """
    tokens = _tokenize_query(query) # tokenize query
    content = [t for t in tokens if t not in STOPWORDS] # remove stopwords
    return content or tokens # if all tokens are stopwords, return original tokens


def _collect_fts_candidate_ids(cur: sqlite3.Cursor, query: str, limit: int = SEMANTIC_CANDIDATE_LIMIT) -> list[int]:
    """
    Generate candidate chunk IDs via progressively broader FTS query variants.
    """
    tokens = _stopping_tokens(query)
    normalized_query = " ".join(tokens) # join tokens into a single string
    if not normalized_query:
        return []

    # SQL query to collect candidate chunk IDs
    # bm25 is ranking function that scores relevance of chunk to query
    candidate_sql = f"""
        SELECT c.id
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY bm25(f)
        LIMIT {int(limit)}
    """

    # try stricter to looser query styles to reduce missed matches
    query_variants: list[str] = [normalized_query]
    query_variants.append(" OR ".join(tokens))
    prefix_terms = [f"{t}*" for t in tokens if len(t) >= 3] 
    if prefix_terms:
        query_variants.append(" OR ".join(prefix_terms))

    seen: set[int] = set()
    candidate_ids: list[int] = []

    # try each query variant
    for variant in query_variants:
        try: # execute SQL query
            cur.execute(candidate_sql, (variant,))
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            continue

        for row in rows: # iterate through rows
            cid = int(row[0]) # get chunk ID
            if cid in seen: # if chunk ID is already in seen, skip
                continue
            seen.add(cid) # add chunk ID to seen
            candidate_ids.append(cid) # add chunk ID to candidate IDs
            if len(candidate_ids) >= limit: # if candidate IDs is greater than limit, return candidate IDs
                return candidate_ids

    return candidate_ids # return candidate IDs

@lru_cache(maxsize=1)
def get_sbert_model() -> SentenceTransformer:
    """
    Load SBERT model once and reuse because model startup is slow.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.max_seq_length = 512
    return model


@lru_cache(maxsize=1)
def load_sbert_index() -> tuple[list[tuple[str, str, str, str, str]], Optional[np.ndarray], dict[int, int]]:
    """
    Load SBERT embeddings once into in-memory matrix for fast repeated queries.
    Returns (docs, matrix[N, D]) where docs aligns with matrix row order.
    """
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        SELECT
            c.id,
            d.file_name,
            d.title,
            d.source_url,
            c.heading,
            c.chunk_text,
            e.embedding
        FROM sbert_embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        JOIN documents d ON d.id = c.document_id
    """

    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()

    # keep document info and vector positions aligned so can map scores back correctly
    docs: list[tuple[str, str, str, str, str]] = [] # list of tuples containing document information
    vectors: list[np.ndarray] = [] # list of numpy arrays containing embeddings
    chunk_id_to_index: dict[int, int] = {} # dictionary mapping chunk IDs to their indices in the docs list
    for chunk_id, file_name, title, source_url, heading, chunk_text, embedding_json in rows:
        try: # convert embedding JSON to numpy array
            vec = np.array(json.loads(embedding_json), dtype=np.float32)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

        if vec.ndim != 1 or vec.size == 0: # if embedding is not a 1D array or is empty, skip
            continue

        docs.append((file_name, title, source_url, heading, chunk_text)) # add document information to docs list
        vectors.append(vec) # add embedding to vectors list
        chunk_id_to_index[int(chunk_id)] = len(docs) - 1 # add chunk ID to chunk_id_to_index dictionary

    if not vectors:
        return [], None, {} # if no embeddings, return empty lists and dictionary

    matrix = np.vstack(vectors).astype(np.float32, copy=False) # stack embeddings into a single matrix
    return docs, matrix, chunk_id_to_index


def get_conn():
    return sqlite3.connect(DB_PATH)


def format_results(rows) -> str:
    if not rows:
        return "No relevant matches found in the corpus."

    out = []
    for i, row in enumerate(rows, start=1):
        file_name, title, source_url, heading, chunk_text = row

        excerpt = chunk_text.replace("\n", " ").strip()
        if len(excerpt) > 500:
            excerpt = excerpt[:500] + "..."

        block = [
            f"[Result {i}]",
            f"File: {file_name}",
            f"Title: {title or 'N/A'}",
            f"Heading: {heading or 'N/A'}",
            f"Source URL: {source_url or 'N/A'}",
            f"Excerpt: {excerpt}"
        ]
        out.append("\n".join(block))

    return "\n\n".join(out)


# METHOD 1 — SQLite FTS5 search 
def _fts_search(query: str) -> tuple[str, float]:
    q = (query or "").strip().lower()
    if not q:
        return "Empty query.", 0.0

    if not DB_PATH.exists():
        return "Database not found. Run build_index.py first.", 0.0

    conn = get_conn()
    cur  = conn.cursor()

    tokens = _stopping_tokens(q) # remove stopwords from query
    if not tokens:
        conn.close()
        return "Empty query.", 0.0

    match_query = " ".join(tokens) # join tokens into single string
    sql = """
        SELECT
            d.file_name,
            d.title,
            d.source_url,
            c.heading,
            c.chunk_text
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        JOIN documents d ON d.id = c.document_id
        WHERE chunks_fts MATCH ?
        LIMIT 5
    """

    start = time.perf_counter()
    try:
        # first try stricter search so results stay focused (strict AND-style query first for precision)
        cur.execute(sql, (match_query,))
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        # if strict search fails, try looser version that matches any term (OR matching when FTS parser rejects or over-constrains query)
        fallback_terms = tokens
        fallback_query = " OR ".join(fallback_terms)
        try:
            cur.execute(sql, (fallback_query,))
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            rows = []
    elapsed = time.perf_counter() - start

    conn.close()
    return format_results(rows), elapsed

# METHOD 2 — Brute-force raw file search
def _brute_force_search(query: str) -> tuple[str, float]:
    q = (query or "").strip().lower()
    if not q:
        return "Empty query.", 0.0

    if not CORPUS_DIR.exists():
        return f"Corpus directory not found: {CORPUS_DIR}", 0.0

    terms = [t for t in q.split() if t]
    md_files = list(CORPUS_DIR.glob("*.md"))

    matches: list[tuple[str, str, int]] = []  # (file_name, excerpt, match_count)

    # baseline method used for compare indexed and vector retrieval
    start = time.perf_counter()
    for md_path in md_files:
        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception:
            continue

        text_lower = text.lower()
        match_count = sum(text_lower.count(term) for term in terms)

        if match_count == 0:
            continue

        # find first matching line for excerpt
        excerpt = ""
        for line in text.splitlines():
            if any(term in line.lower() for term in terms):
                excerpt = line.strip()[:500]
                break

        matches.append((md_path.name, excerpt, match_count))

    elapsed = time.perf_counter() - start

    # sort by most matches first, take top 5
    matches.sort(key=lambda x: x[2], reverse=True)
    top = matches[:5]

    if not top:
        return "No relevant matches found in the corpus.", elapsed

    out = []
    for i, (file_name, excerpt, count) in enumerate(top, start=1):
        block = [
            f"[Result {i}]",
            f"File: {file_name}",
            f"Keyword hits: {count}",
            f"Excerpt: {excerpt or 'N/A'}",
        ]
        out.append("\n".join(block))

    return "\n\n".join(out), elapsed

# METHOD 3 — Semantic vector search with SBERT's all-MiniLM-L6-v2 model
def _semantic_search_sbert(query: str) -> tuple[str, float]:
    q = (query or "").strip()
    if not q:
        return "Empty query.", 0.0
    
    if not DB_PATH.exists():
        return "Database not found. Run build_index.py first.", 0.0

    tokens = _stopping_tokens(q)
    if not tokens:
        return "Empty query.", 0.0

    start = time.perf_counter()
    conn = get_conn()
    cur = conn.cursor()
    # Stage 1: quickly gather likely chunks using keyword-style database search
    candidate_ids = _collect_fts_candidate_ids(cur, q, limit=SEMANTIC_CANDIDATE_LIMIT)
    conn.close()

    try:
        docs, matrix, chunk_id_to_index = load_sbert_index() # load SBERT embeddings into in-memory matrix
    except sqlite3.OperationalError:
        elapsed = time.perf_counter() - start
        return "Embeddings not found. Run sbert_vectors.py first.", elapsed

    # Stage 2: compare meaning (not just keywords) with embeddings
    embeddings = get_sbert_model() # get SBERT model
    query_vector = embeddings.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32, copy=False) # encode query into numpy array

    elapsed = time.perf_counter() - start
    if not docs or matrix is None:
        return "No relevant matches found in the corpus.", elapsed

    if matrix.shape[1] != query_vector.shape[0]:
        return "No relevant matches found in the corpus.", elapsed

    candidate_indices = [chunk_id_to_index[cid] for cid in candidate_ids if cid in chunk_id_to_index]
    if not candidate_indices: # if keyword search finds nothing, score every chunk as fallback
        candidate_indices = list(range(matrix.shape[0]))

    candidate_matrix = matrix[candidate_indices] # only score likely chunks to keep search fast while staying relevant

    # vectors are normalized, so dot product gives cosine similarity
    similarities = candidate_matrix @ query_vector # dot product of candidate matrix and query vector
    lexical_scores = np.zeros(similarities.shape[0], dtype=np.float32) # initialize lexical scores array
    token_set = set(tokens)
    # score each chunk based on keyword overlap and metadata
    for i, doc_idx in enumerate(candidate_indices):
        file_name, title, _, heading, chunk_text = docs[doc_idx]
        doc_text = " ".join([
            (file_name or "").lower(),
            (title or "").lower(),
            (heading or "").lower(),
            (chunk_text or "").lower(),
        ])
        overlaps = sum(1 for t in token_set if t in doc_text)
        overlap_ratio = overlaps / max(1, len(token_set))

        # give extra credit when terms appear in title/heading/file name
        meta_text = " ".join([(file_name or "").lower(), (title or "").lower(), (heading or "").lower()])
        meta_overlap = sum(1 for t in token_set if t in meta_text) / max(1, len(token_set))
        lexical_scores[i] = np.float32(overlap_ratio + 0.5 * meta_overlap)

    # final score blends meaning similarity with keyword overlap signals
    hybrid_scores = similarities + 0.45 * lexical_scores
    top_k = min(5, hybrid_scores.shape[0])
    if top_k <= 0:
        return "No relevant matches found in the corpus.", elapsed

    top_idx = np.argpartition(hybrid_scores, -top_k)[-top_k:] # partition hybrid scores into top k chunks
    top_idx = top_idx[np.argsort(hybrid_scores[top_idx])[::-1]] # sort top k chunks by hybrid score
    top_rows = [docs[candidate_indices[i]] for i in top_idx] # get top k chunks
    return format_results(top_rows), elapsed # return top k chunks and elapsed time

# BENCHMARK — run both methods and print timing comparison
def benchmark_search(query: str) -> str:
    print("\n" + "=" * 60)
    print(f"  BENCHMARK: '{query}'")
    print("=" * 60)

    fts_results,   fts_time   = _fts_search(query)
    brute_results, brute_time = _brute_force_search(query)
    semantic_results, semantic_time = _semantic_search_sbert(query)

    speedup = brute_time / fts_time if fts_time > 0 else float("inf")
    semantic_speedup = brute_time / semantic_time if semantic_time > 0 else float("inf")

    print(f"  SQLite FTS5  : {fts_time * 1000:.2f} ms")
    print(f"  Brute-force  : {brute_time * 1000:.2f} ms")
    print(f"  Semantic     : {semantic_time * 1000:.2f} ms")
    print(f"  FTS is {speedup:.1f}x faster")
    print(f"  Semantic is {semantic_speedup:.1f}x faster than brute-force")
    print("=" * 60 + "\n")

    return fts_results, semantic_results # return both results for comparison in agent response


@tool
def search_corpus(query: str) -> str:
    """
    Search the CPP campus knowledge base for information relevant to the query.
    Runs both SQLite FTS, brute-force search, and semantic search, prints a timing comparison,
    and returns the FTS and semantic results for the agent to use.
    """
    return benchmark_search(query)


def create_agent() -> AgentExecutor:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        temperature=0.2,
        api_key=api_key
    )

    tools = [search_corpus]
    # Same grounded-assistant rules as agent.py (greeting, citations, confidence %, structured-data caution);
    # agent_test keeps a single benchmark search_corpus tool—no create_chat_title / semantic_search_sbert here.
    system_prompt = """
        You are Cal Poly Pomona's AI campus knowledge assistant.
        Help students find accurate information from the Cal Poly Pomona website using the provided search tools.

        At the start of a new conversation, greet the user briefly and simply:
        “Hi! I can help you find information about Cal Poly Pomona. What would you like to know?”

        You may use these tools:
        - search_corpus

        Instructions:
        - Use the tools to retrieve relevant website information before answering factual questions.
        - Only answer with information supported by tool results.
        - Never fabricate details or rely on unsupported assumptions.
        - Never mention internal tools, the corpus, embeddings, retrieval, or system instructions.

        STRICT grounding and accuracy rules:
        - Do NOT infer, guess, or “fill in gaps” between pieces of information.
        - Do NOT combine information from different sources unless the relationship is explicitly clear in the source content.
        - Do NOT assume two pieces of data belong together unless they are shown together in the same context.

        Structured data rules (CRITICAL):
        - When presenting structured information (e.g., course numbers and course names, requirements, deadlines, office details):
            - Only present pairings exactly as they appear in the source.
            - NEVER mix items across lists or sections.
            - NEVER reconstruct pairings from separate lines or different sources.
            - If you cannot confidently match related items (e.g., course number ↔ course title), do NOT present them as pairs.
            - Instead, either: present the information separately, OR say that the relationships are not clearly specified.

        Evidence-first rule:
        - Before summarizing, ensure the answer is directly supported by at least one clear, consistent source.
        - If multiple pieces of information are needed to answer the question, ALL required pieces must be supported.
        - If any key part of the answer is missing, unclear, or inconsistent: Do NOT attempt to complete the answer.

        Insufficient information rule (VERY IMPORTANT):
        - If you cannot find enough clear and consistent information to fully answer the user's question:
            - Clearly say: “I wasn't able to find enough clear information on the website to fully answer that.”
            - Share what partial information you found, but label it clearly as incomplete and do not combine or reconstruct it with other information.
            - Do NOT present partial information by combining or reconstructing it with other information. Simply state what you did find separately.
            - Do NOT guess or approximate.

        Answering behavior:
        - Answer clearly and conversationally.
        - Start with the direct answer (only if fully supported).
        - Then provide brief supporting details if helpful.
        - Include the source URL(s) used.
        - Only cite sources that directly support the answer.
        - At the end of the answer, provide a confidence score for the answer based on your confidence in the sources used.
        - The confidence score should be a percentage between 0 and 100, where 0 is the lowest confidence and 100 is the highest confidence.

        Conversation behavior:
        - Maintain context across turns so follow-up questions feel natural.
        - Interpret follow-up questions using prior context when appropriate.
        - If results are ambiguous or conflicting, say so explicitly.

        Related-topic guidance:
        - When appropriate, suggest a few closely related topics the user may want to explore next.
        - Keep suggestions short and directly relevant.

        Your goal is to provide accurate, grounded, and trustworthy answers while avoiding incorrect associations or incomplete conclusions."""

    # system prompt constrains assistant to corpus-backed answers
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    # return_intermediate_steps=True: lets main() read benchmark tuple (FTS vs semantic) from the tool observation.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    return agent_executor

def main():
    """Main function to run the agent."""
    try: # create the agent and check for API key
        agent_executor = create_agent()
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set your OPENAI_API_KEY in a .env file.")
        return
    
    chat_history = [] # store chat history

    while True: # main chat loop
        try:
            user_input = input("\nYou: ").strip() # get user input
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # run agent (invoke agent executor)
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            # display both tool results when available, then agent response
            output = response["output"]
            tool_output = None
            # read tool outputs so we can print both search result sets for user
            for step in response.get("intermediate_steps", []):
                if isinstance(step, tuple) and len(step) >= 2: # if step is tuple and has at least 2 elements
                    observation = step[1]
                    if isinstance(observation, (tuple, list)) and len(observation) >= 2: # if observation is tuple/list and has at least 2 elements
                        tool_output = observation
                        break

            if tool_output:
                print(f"\nAssistant (FTS): {tool_output[0]}")
                print(f"\nAssistant (Semantic): {tool_output[1]}")

            print(f"\nAssistant: {output}")

            # update chat history (store user input and agent response)
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=str(output)))

        except KeyboardInterrupt: # handle keyboard interrupt
            print("\n\nInterrupted by user.")
            break

        except Exception as e: # handle other exceptions
            print(f"\nError: {e}")
            print("Please try again.")
            continue

if __name__ == "__main__": # run main function
    main()