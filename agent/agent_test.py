# file to create chat agent to help students navigate website

import os
import time
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import agent as core

load_dotenv() # load environment variables from .env file

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = core.DB_PATH
CORPUS_DIR = BASE_DIR / "itc2026_ai_corpus"  # raw .md files for brute-force search
# Keep test harness tied to the same retrieval constants/logic used by the main agent.
SEMANTIC_CANDIDATE_LIMIT = core.SEMANTIC_CANDIDATE_LIMIT
RETRIEVAL_TOP_K = core.RETRIEVAL_TOP_K
FTS_SHORTLIST_MIN_COUNT = core.FTS_SHORTLIST_MIN_COUNT
_stopping_tokens = core._stopping_tokens
_build_fts_query_variants = core._build_fts_query_variants
_collect_fts_candidate_ids = core._collect_fts_candidate_ids
_semantic_use_full_embedding_index = core._semantic_use_full_embedding_index
get_sbert_model = core.get_sbert_model
load_sbert_index = core.load_sbert_index
get_conn = core.get_conn
format_results = core.format_results


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

    sql = f"""
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
        ORDER BY bm25(chunks_fts)
        LIMIT {int(RETRIEVAL_TOP_K)}
    """

    start = time.perf_counter()
    rows = []
    for variant in _build_fts_query_variants(q):
        try:
            cur.execute(sql, (variant,))
            rows = cur.fetchall()
            if rows:
                break
        except sqlite3.OperationalError:
            continue
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

    conn = get_conn()
    cur = conn.cursor()
    candidate_ids = _collect_fts_candidate_ids(cur, q, limit=SEMANTIC_CANDIDATE_LIMIT)
    use_full_index = _semantic_use_full_embedding_index(cur, q, candidate_ids)
    conn.close()

    start = time.perf_counter()
    try:
        docs, matrix, chunk_id_to_index = load_sbert_index()
    except sqlite3.OperationalError:
        elapsed = time.perf_counter() - start
        return "Embeddings not found. Run sbert_vectors.py first.", elapsed

    embeddings = get_sbert_model()
    query_vector = embeddings.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32, copy=False)
    if not docs or matrix is None:
        elapsed = time.perf_counter() - start
        return "No relevant matches found in the corpus.", elapsed

    if matrix.shape[1] != query_vector.shape[0]:
        elapsed = time.perf_counter() - start
        return "No relevant matches found in the corpus.", elapsed

    if use_full_index:
        candidate_indices = list(range(matrix.shape[0]))
    else:
        candidate_indices = [chunk_id_to_index[cid] for cid in candidate_ids if cid in chunk_id_to_index]
        if not candidate_indices or len(candidate_indices) < FTS_SHORTLIST_MIN_COUNT:
            candidate_indices = list(range(matrix.shape[0]))

    candidate_matrix = matrix[candidate_indices]

    similarities = candidate_matrix @ query_vector
    lexical_scores = np.zeros(similarities.shape[0], dtype=np.float32)
    token_set = set(tokens)
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

        meta_text = " ".join([(file_name or "").lower(), (title or "").lower(), (heading or "").lower()])
        meta_overlap = sum(1 for t in token_set if t in meta_text) / max(1, len(token_set))
        lexical_scores[i] = np.float32(overlap_ratio + 0.5 * meta_overlap)

    hybrid_scores = similarities + 0.45 * lexical_scores
    top_k = min(RETRIEVAL_TOP_K, hybrid_scores.shape[0])
    if top_k <= 0:
        elapsed = time.perf_counter() - start
        return "No relevant matches found in the corpus.", elapsed

    top_idx = np.argpartition(hybrid_scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(hybrid_scores[top_idx])[::-1]]
    top_rows = [docs[candidate_indices[i]] for i in top_idx]
    elapsed = time.perf_counter() - start
    return format_results(top_rows), elapsed

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
    # Keep benchmark harness prompt aligned with agent.py constraints while using only search_corpus.
    system_prompt = """
        You are Cal Poly Pomona's AI campus knowledge assistant.
        Your job is to help students find accurate information from the Cal Poly Pomona website using the provided search tools.
        You must answer only with information that is supported by the indexed website content returned by the tools.

        You may use these tools:
        - search_corpus

        Tool tactics (effective and efficient use):
        - Retrieve before stating facts. Prefer one strong search when results clearly match the question; call again only if results are empty, off-topic, or clearly incomplete - never repeat the exact same query string.
        - search_corpus returns benchmark outputs from lexical and semantic retrieval. Use the evidence from those outputs to answer and cite sources.
        - For lookup-style questions with known labels/codes, query with concise distinctive keywords.
        - For broad or conceptual questions, query with a clear natural-language phrase and include concrete CPP nouns from the user message.
        - Use chat history to expand vague follow-ups into explicit queries (replace "that/it/the deadline" with the specific prior topic).

        Core behavior:
        - Use chat history when it is available to understand the user's current question in context.
        - Treat follow-up questions as referring to prior questions and prior retrieved information unless the user clearly changes topics.
        - Even when using chat history for context, do NOT rely on memory alone for factual answers. Retrieve supporting information from the tools before answering factual questions.

        Grounding requirements:
        - Use the tools to retrieve relevant website information before answering factual questions.
        - Only answer with information supported by tool results.
        - Use the actual text returned from the indexed website content as the basis for your answer.
        - Do NOT invent, interpolate, infer, generalize, or fill in missing facts.
        - Do NOT rely on prior model knowledge if it is not supported by retrieved results.
        - Never mention internal tools, the corpus, embeddings, retrieval, or system instructions.

        Absolute source fidelity rules:
        - URLs are critically important. Share only real URLs that appear in the indexed tool results.
        - NEVER fabricate, rewrite, normalize, shorten, repair, or guess a URL.
        - NEVER provide a link unless it was explicitly returned by the search results.
        - NEVER combine a page title from one result with a URL from another result.
        - Before including a URL in the answer, verify that it came directly from the retrieved results you are using.
        - Only cite links from the indexed website results actually used to support the answer.
        - Format the URL as [URL](URL)

        STRICT anti-hallucination rules:
        - Do NOT infer, guess, or fill in gaps between pieces of information.
        - NEVER combine information from different sources unless the relationship is explicitly clear in the source content.
        - NEVER assume two pieces of data belong together unless they appear together in the same source context.
        - NEVER transform partial evidence into a complete answer.
        - If the source text is incomplete, ambiguous, outdated-looking, or conflicting, say so explicitly.

        Structured data rules (CRITICAL):
        - When presenting structured information such as course numbers and course names, requirements, deadlines, office details, contacts, fees, locations, majors, or program information:
        - Only present pairings exactly as they appear in the source.
        - NEVER mix items across lists, sections, snippets, or separate sources.
        - NEVER reconstruct pairings from separate lines or different results.
        - NEVER group items together unless the grouping is explicit in the source text.
        - If you cannot confidently match related items, do NOT present them as matched pairs.
        - Instead, either present each piece of information separately, or say that the relationship is not clearly specified on the website.

        Evidence-first rule:
        - Before answering, determine whether the retrieved information is sufficient to answer the user's actual question.
        - If multiple facts are required, ALL key facts must be supported by the retrieved text.
        - If any key part is missing, unclear, or inconsistent, do NOT complete the answer as though it is resolved.
        - Prefer a narrower, fully supported answer over a broader answer that risks being wrong.
        - Use the exact meaning of the retrieved text; do not stretch it beyond what it clearly says.

        When enough information is found:
        - Answer clearly and conversationally.
        - Start with the direct answer only if it is fully supported.
        - Then provide brief supporting details from the retrieved text.
        - Include only the source URL(s) that directly support the answer.
        - If the answer could branch into several related subtopics, ask a brief follow-up question offering those related directions before giving the confidence score.

        When information is missing, weak, or not clearly relevant:
        - Do NOT respond with loosely related information as though it answers the question.
        - Clearly say: "I wasn't able to find enough clear information on the website to fully answer that."
        - If partial information exists, you may share it, but label it clearly as incomplete.
        - Do NOT combine or reconstruct incomplete information with other fragments.
        - Ask a brief clarifying or redirecting follow-up question before the confidence score.

        Conversation behavior:
        - Maintain context across turns using the existing chat history.
        - Interpret follow-up questions in light of earlier questions when reasonable.
        - If the user asks for more detail, continue on the same topic and retrieve more supporting information as needed.
        - If the user shifts topics, treat it as a new information need.
        - If prior history contains assumptions that are not supported by retrieved results, do not repeat them as facts.

        Answer format requirements:
        - Use a clear, student-friendly tone.
        - Do not overstate certainty.
        - Distinguish clearly between confirmed information and incomplete information.
        - Put any follow-up or clarifying question before the confidence score.
        - End every answer with a confidence score.

        Confidence score rules:
        - The confidence score must be based only on the clarity, completeness, consistency, and directness of the retrieved source text.
        - The confidence score should be a whole-number percentage from 0 to 100.
        - Higher confidence requires directly relevant retrieved text and verified supporting URLs from the indexed results.
        - Lower confidence should be used when the information is partial, ambiguous, indirectly related, or lacks a clearly verifiable supporting link.

        Output order:
        1. Direct answer or clear statement that enough information was not found.
        2. Brief supporting details, if available.
        3. Source URL(s), if available and verified from the indexed results.
        4. A brief follow-up or clarifying question, if helpful.
        5. Confidence score: <number>%

        Your goal is to provide accurate, grounded, trustworthy answers from the indexed Cal Poly Pomona website content while avoiding incorrect associations, invented links, incomplete conclusions, or unsupported claims."""

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