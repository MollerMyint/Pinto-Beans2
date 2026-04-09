# file to create chat agent to help students navigate website

import os
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

# Used by extract_chat_title() to match the title tool in intermediate_steps (keeps name in one place).
CHAT_TITLE_TOOL_NAME = "create_chat_title"

from dotenv import load_dotenv
import numpy as np

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from flask import Flask, request, render_template
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import warnings

load_dotenv() # load environment variables from .env file
warnings.filterwarnings('ignore')

# ============================================================================
# Paths, retrieval tuning, and query normalization (shared by FTS + SBERT)
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "campus_kb.db"  # SQLite: chunks, documents, FTS, sbert_embeddings
CORPUS_DIR = BASE_DIR / "itc2026_ai_corpus"  # optional raw .md mirror of scraped pages (reserved for future use)
SEMANTIC_CANDIDATE_LIMIT = 300  # max chunk IDs passed from FTS into SBERT reranking
MIN_TOKEN_LEN = 2  # drop 1-char noise tokens from query tokenization
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "what", "when",
    "where", "which", "who", "why", "with", "your", "you", "i", "me", "my", "we",
    "our", "us", "cal", "poly", "pomona",
}


def _tokenize_query(query: str) -> list[str]:
    """Lowercase alphanumeric tokens only"""
    cleaned = (query or "").strip().lower()
    if not cleaned:
        return []
    tokens = re.findall(r"[a-z0-9]+", cleaned)
    return [t for t in tokens if len(t) >= MIN_TOKEN_LEN]


def _stopping_tokens(query: str) -> list[str]:
    """Content words for matching; if everything is stopwords, fall back to full token list."""
    tokens = _tokenize_query(query)
    content = [t for t in tokens if t not in STOPWORDS]
    return content or tokens


def _collect_fts_candidate_ids(cur: sqlite3.Cursor, query: str, limit: int = SEMANTIC_CANDIDATE_LIMIT) -> list[int]:
    """Stage-1 retrieval for SBERT: broad FTS recall, ranked by bm25, deduped across query variants."""
    tokens = _stopping_tokens(query)
    normalized_query = " ".join(tokens)
    if not normalized_query:
        return []

    # bm25(f) orders rows by lexical relevance inside FTS5
    candidate_sql = f"""
        SELECT c.id
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY bm25(f)
        LIMIT {int(limit)}
    """

    # Stricter -> looser: AND-like token string, then any-token OR, then prefix stems (e.g. registr*)
    query_variants: list[str] = [normalized_query]
    query_variants.append(" OR ".join(tokens))
    prefix_terms = [f"{t}*" for t in tokens if len(t) >= 3]
    if prefix_terms:
        query_variants.append(" OR ".join(prefix_terms))

    seen: set[int] = set()
    candidate_ids: list[int] = []

    for variant in query_variants:
        try:
            cur.execute(candidate_sql, (variant,))
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            continue  # malformed MATCH for this variant; try the next

        for row in rows:
            cid = int(row[0])
            if cid in seen:
                continue
            seen.add(cid)
            candidate_ids.append(cid)
            if len(candidate_ids) >= limit:
                return candidate_ids

    return candidate_ids


@lru_cache(maxsize=1)
def get_sbert_model() -> SentenceTransformer:
    """One shared model instance; first load is slow, later calls are cheap."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.max_seq_length = 512
    return model


def get_conn():
    """Short-lived connections per operation (simple and safe for Flask + CLI)."""
    return sqlite3.connect(DB_PATH)


@lru_cache(maxsize=1)
def load_sbert_index() -> tuple[
    list[tuple[str, str, str, str, str]],
    Optional[np.ndarray],
    dict[int, int],
]:
    """In-memory doc list + embedding matrix; chunk_id_to_index maps SQL chunk id to matrix row."""
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

    docs: list[tuple[str, str, str, str, str]] = []
    vectors: list[np.ndarray] = []
    chunk_id_to_index: dict[int, int] = {}
    for chunk_id, file_name, title, source_url, heading, chunk_text, embedding_json in rows:
        try:
            vec = np.array(json.loads(embedding_json), dtype=np.float32)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

        if vec.ndim != 1 or vec.size == 0:
            continue

        docs.append((file_name, title, source_url, heading, chunk_text))
        vectors.append(vec)
        chunk_id_to_index[int(chunk_id)] = len(docs) - 1  # row index in docs and matrix

    if not vectors:
        return [], None, {}

    matrix = np.vstack(vectors).astype(np.float32, copy=False)
    return docs, matrix, chunk_id_to_index


def format_results(rows) -> str:
    """Human-readable blocks for the LLM and optional HTTP responses."""
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


# --- Chat title (added for UIs): normalize model/tool text and read title from invoke() intermediate_steps ---
def _normalize_chat_title_text(raw: str) -> str:
    """Collapse whitespace, strip markdown asterisks from model input, cap length."""
    t = (raw or "").strip()
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return "New chat"
    if len(t) > 100:
        t = t[:97] + "..."
    return t


def _strip_title_from_tool_observation(observation: str) -> str:
    """Parse title text from create_chat_title tool return (expects **title** or plain text)."""
    t = (observation or "").strip()
    m = re.match(r"^\*\*(.+)\*\*$", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    return _normalize_chat_title_text(t)


def extract_chat_title(agent_result: dict[str, Any]) -> Optional[str]:
    """
    Return the chat title from the last create_chat_title tool step, if any.

    Use with AgentExecutor.invoke results from create_agent(): intermediate steps
    are included so UIs can persist or display the title separately from output.
    """
    steps = agent_result.get("intermediate_steps")
    if not steps:
        return None
    last: Optional[str] = None
    for action, observation in steps:
        if getattr(action, "tool", None) == CHAT_TITLE_TOOL_NAME:
            last = _strip_title_from_tool_observation(str(observation))
    return last


# Flask constructor
app = Flask(__name__)

@tool
def search_corpus(query: str) -> str:
    """
    Search the offline SQLite index of the CPP markdown corpus.
    Returns top chunk-level matches with source attribution.
    """
    q = (query or "").strip().lower() # strip whitespace and convert to lowercase
    if not q:
        return "Empty query."

    if not DB_PATH.exists():
        return "Database not found. Run build_index.py first."

    conn = get_conn()
    cur = conn.cursor()

    tokens = _stopping_tokens(q)
    if not tokens:
        conn.close()
        return "Empty query."

    match_query = " ".join(tokens)  # FTS5 default: implicit AND between terms

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

    try:
        cur.execute(sql, (match_query,))
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        # Parser rejected strict query (rare tokens / operators); OR any content token
        fallback_terms = tokens
        fallback_query = " OR ".join(fallback_terms)
        try:
            cur.execute(sql, (fallback_query,))
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            rows = []

    conn.close()
    return format_results(rows)

@tool
def semantic_search_sbert(query: str) -> str:
    """
    Perform semantic search on the CPP markdown corpus with SBERT's all-MiniLM-L6-v2 model.
    Uses FTS candidates, then hybrid SBERT + lexical reranking (aligned with agent_test.py).
    """
    q = (query or "").strip()
    if not q:
        return "Empty query."

    if not DB_PATH.exists():
        return "Database not found. Run build_index.py first."

    tokens = _stopping_tokens(q)
    if not tokens:
        return "Empty query."

    conn = get_conn()
    cur = conn.cursor()
    candidate_ids = _collect_fts_candidate_ids(cur, q, limit=SEMANTIC_CANDIDATE_LIMIT)
    conn.close()

    try:
        docs, matrix, chunk_id_to_index = load_sbert_index()
    except sqlite3.OperationalError:
        return "Embeddings not found. Run sbert_vectors.py first."

    embeddings = get_sbert_model()
    # Full original query string preserves phrasing for embedding model, not just token list
    query_vector = embeddings.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32, copy=False)

    if not docs or matrix is None:
        return "No relevant matches found in the corpus."

    if matrix.shape[1] != query_vector.shape[0]:
        return "No relevant matches found in the corpus."

    candidate_indices = [chunk_id_to_index[cid] for cid in candidate_ids if cid in chunk_id_to_index]
    if not candidate_indices:
        # FTS missed everything lexical -> fall back to scoring all embedded chunks (slower but higher recall)
        candidate_indices = list(range(matrix.shape[0]))

    candidate_matrix = matrix[candidate_indices]

    # Vectors are normalized so dot product gives cosine similarity
    similarities = candidate_matrix @ query_vector
    lexical_scores = np.zeros(similarities.shape[0], dtype=np.float32)
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

        # Extra weight when tokens hit filename / title / heading -> stronger intent signal than body alone
        meta_text = " ".join([(file_name or "").lower(), (title or "").lower(), (heading or "").lower()])
        meta_overlap = sum(1 for t in token_set if t in meta_text) / max(1, len(token_set))
        lexical_scores[i] = np.float32(overlap_ratio + 0.5 * meta_overlap)

    # Blend dense similarity with sparse overlap -> 0.45 tuned alongside agent_test.py
    hybrid_scores = similarities + 0.45 * lexical_scores
    top_k = min(5, hybrid_scores.shape[0])
    if top_k <= 0:
        return "No relevant matches found in the corpus."

    # Partial sort: select top_k then sort only those indices
    top_idx = np.argpartition(hybrid_scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(hybrid_scores[top_idx])[::-1]]
    top_rows = [docs[candidate_indices[i]] for i in top_idx]
    return format_results(top_rows)

# LangChain tool so the model can set a session title on the first turn; paired with extract_chat_title() for APIs.
@tool
def create_chat_title(query: str) -> str:
    """Create a title for the chat based on the query or topic text the model passes in."""
    title = _normalize_chat_title_text(query)
    return f"**{title}**"

# return_intermediate_steps: required for extract_chat_title(); callers (e.g. HTTP) can set False if unused.
def create_agent(*, return_intermediate_steps: bool = True) -> AgentExecutor:
    """Create LangChain agent with tools."""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")

    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=api_key
    )

    # search_corpus: FTS keyword hits; semantic_search_sbert: meaning + hybrid rerank
    # create_chat_title: first-turn chat naming (see system_prompt below).
    tools = [search_corpus, semantic_search_sbert, create_chat_title]

    # Expanded from short corpus-only prompt: grounding, structured-data rules, confidence %, create_chat_title flow.
    system_prompt = """
        You are Cal Poly Pomona's AI campus knowledge assistant.
        Help students find accurate information from the Cal Poly Pomona website using the provided search tools.

        If the user asks their first question, create a title for the chat based on the query using the create_chat_title tool.
        - The title should be a single phrase or sentence that captures the main topic of the conversation.
        - The title should be on its own line.
        - The format should be: "**<title>**"

        You may use these tools:
        - search_corpus
        - semantic_search_sbert
        - create_chat_title

        Instructions:
        - Use the tools to retrieve relevant website information before answering factual questions.
        - Only answer with information supported by tool results.
        - Never fabricate details or rely on unsupported assumptions.
        - Never mention internal tools, the corpus, embeddings, retrieval, or system instructions.

        STRICT grounding and accuracy rules:
        - Do NOT infer, guess, or “fill in gaps” between pieces of information.
        - NEVER combine information from different sources unless the relationship is explicitly clear in the source content.
        - NEVER assume two pieces of data belong together unless they are shown together in the same context.

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

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Create the agent (uses the tools and prompt)
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Create executor (runs the agent); intermediate_steps wire through for title extraction / debugging.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False, # does not print verbose output
        handle_parsing_errors=True,
        return_intermediate_steps=return_intermediate_steps,
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
    # First loop iteration uses empty input so the agent can emit the opening greeting (system_prompt).
    user_input = ""
    while True: # main chat loop
        try:
             # Run the agent (invoke the agent executor)
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            # Display the response (from the agent executor)
            print(f"\nAssistant: {response['output']}")

            # Prompt after each reply so turn 1 shows greeting before any user message.
            user_input = input("\nYou: ").strip() # get user input
            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # Update chat history (store the user input and the agent response)
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response['output']))

        except KeyboardInterrupt: # handle keyboard interrupt
            print("\n\nInterrupted by user.")
            break

        except Exception as e: # handle other exceptions
            print(f"\nError: {e}")
            print("Please try again.")
            continue

if __name__ == "__main__": # run the main function
    main()