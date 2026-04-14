# file to create chat agent to help students navigate website

import os
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
import numpy as np

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from flask import Flask, request, render_template
from functools import lru_cache
# temp deletion
# from sentence_transformers import SentenceTransformer
import warnings

load_dotenv()  # load environment variables from .env file
warnings.filterwarnings('ignore')

# Used by extract_chat_title() to match title tool in intermediate_steps (keeps name in one place)
CHAT_TITLE_TOOL_NAME = "create_chat_title"

# ============================================================================
# Paths, retrieval tuning, and query normalization (shared by FTS + SBERT)
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "campus_kb.db"  # SQLite: chunks, documents, FTS, sbert_embeddings
CORPUS_DIR = BASE_DIR / "itc2026_ai_corpus"  # optional raw .md mirror of scraped pages (reserved for future use)
SEMANTIC_CANDIDATE_LIMIT = 300  # max chunk IDs passed from FTS into SBERT reranking
RETRIEVAL_TOP_K = 15  # chunks returned to the model per search_* tool call (recall vs context size)

# When lexical recall looks weak, run SBERT cosine over every embedded chunk instead of FTS candidates only
FTS_SHORTLIST_MIN_COUNT = 40  # fewer unique FTS candidates than this → full dense index

# SQLite FTS5 bm25 (chunks_fts): more negative = better here
# if best strict AND hit is above this (weaker / less negative), treat lexical match as low-confidence and score full embedding index
FTS_BM25_WEAK_IF_ABOVE = -5.5
MIN_TOKEN_LEN = 2  # drop 1-char noise tokens from query tokenization
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "what", "when",
    "where", "which", "who", "why", "with", "your", "you", "i", "me", "my", "we",
    "our", "us",
}


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """Deduplicate strings while preserving insertion order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = (item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _sliding_phrases(tokens: list[str], *, min_n: int = 2, max_n: int = 3, max_phrases: int = 6) -> list[str]:
    """
    Build quoted FTS5 phrase queries from adjacent tokens.
    (Example: tokens ["financial", "aid", "deadline"] -> phrase query "financial aid".)
    Uses short n-grams to preserve meaningful multi-word concepts without hardcoding topics.
    """
    phrases: list[str] = []
    if len(tokens) < min_n:
        return phrases
    # build quoted FTS5 phrase queries from adjacent tokens
    for n in range(max_n, min_n - 1, -1):
        if len(tokens) < n:
            continue
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i + n]).strip()
            if phrase:
                phrases.append(f"\"{phrase}\"")
            if len(phrases) >= max_phrases:
                return phrases
    return phrases


def _build_fts_query_variants(query: str) -> list[str]:
    """
    Generate generic, topic-agnostic FTS query variants from strict to broad.
    ("Topic-agnostic" means this logic does not depend on campus-specific keywords/rules.)

    Strategy:
    1) Strict token AND-style query.
    2) Phrase variants from adjacent token n-grams.
    3) Leave-one-out relaxed variants for long queries.
       (Create versions that drop one term at a time to recover from one noisy token.)
    4) Token OR variant.
    5) Prefix OR variant to catch stems.
    """
    tokens = _stopping_tokens(query)
    if not tokens:
        return []

    variants: list[str] = []
    # 1st variant intentionally mirrors old behavior (implicit AND between terms), so we only broaden if strict matching under-retrieves
    normalized_query = " ".join(tokens)
    variants.append(normalized_query)

    # phrase matching captures concepts represented as multi-word labels
    variants.extend(_sliding_phrases(tokens))

    # relax strict matching for longer queries by dropping one token at a time (avoids "all terms required" behavior)
    if len(tokens) >= 4:
        for i in range(len(tokens)):
            relaxed = [t for j, t in enumerate(tokens) if j != i]
            if len(relaxed) >= 2:
                variants.append(" ".join(relaxed))

    variants.append(" OR ".join(tokens))

    # add prefix variants for longer terms (e.g., "cs" → "cs*")
    prefix_terms = [f"{t}*" for t in tokens if len(t) >= 3]
    if prefix_terms:
        variants.append(" OR ".join(prefix_terms))

    return _dedupe_keep_order(variants)


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
    if not tokens:
        return []

    # bm25(chunks_fts) orders rows by lexical relevance (table name required, not alias)
    candidate_sql = f"""
        SELECT c.id
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY bm25(chunks_fts)
        LIMIT {int(limit)}
    """

    # Stricter to looser generic variants (topic-agnostic)
    query_variants = _build_fts_query_variants(query)

    seen: set[int] = set()
    candidate_ids: list[int] = []

    for variant in query_variants:
        try:
            cur.execute(candidate_sql, (variant,))
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            continue  # malformed MATCH for this variant; try next

        for row in rows:
            cid = int(row[0])
            if cid in seen:
                continue
            seen.add(cid)
            candidate_ids.append(cid)
            if len(candidate_ids) >= limit:
                return candidate_ids

    return candidate_ids


def _fts_strict_best_bm25(cur: sqlite3.Cursor, normalized_query: str) -> Optional[float]:
    """Best bm25 among strict AND-style matches; None if no match or FTS error."""
    if not normalized_query:
        return None
    sql = """
        SELECT bm25(chunks_fts)
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY bm25(chunks_fts)
        LIMIT 1
    """
    try:
        cur.execute(sql, (normalized_query,))
        row = cur.fetchone()
    except sqlite3.OperationalError:
        return None
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return None


def _semantic_use_full_embedding_index(cur: sqlite3.Cursor, query: str, candidate_ids: list[int]) -> bool:
    """
    True when the FTS-derived candidate pool is too small or lexically weak, so dense search should score all embedded chunks instead of the shortlist.
    """
    # too few lexical candidates tends mean recall is fragile for paraphrased questions
    if len(candidate_ids) < FTS_SHORTLIST_MIN_COUNT:
        return True
    normalized = " ".join(_stopping_tokens(query))
    best = _fts_strict_best_bm25(cur, normalized)
    if best is None:  # shortlist came from looser variants only (BM25 pool may miss paraphrases)
        return True
    # bm25 is better when more negative in FTS5; scores above threshold considered weak
    return best > FTS_BM25_WEAK_IF_ABOVE

    # @lru_cache(maxsize=1)
    # def get_sbert_model() -> SentenceTransformer:
    """One shared model instance; first load is slow, later calls are cheap."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.max_seq_length = 512
    return model


def get_conn():
    """Short-lived connections per operation (simple and safe for Flask + CLI)."""
    return sqlite3.connect(DB_PATH)

    # @lru_cache(maxsize=1)
    # def load_sbert_index() -> tuple[
    list[tuple[str, str, str, str, str]],
    Optional[np.ndarray],
    dict[int, int],
    # ]:
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
    q = (query or "").strip().lower()  # strip whitespace and convert to lowercase
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

    rows = []
    query_variants = _build_fts_query_variants(q)
    # try strict to broad variants until one returns hits; avoids empty responses if malformed token/rare phrase breaks strict MATCH behavior
    for variant in query_variants:
        try:
            cur.execute(sql, (variant,))
            rows = cur.fetchall()
            if rows:
                break
        except sqlite3.OperationalError:
            continue

    conn.close()
    return format_results(rows)

    # @tool
    # def semantic_search_sbert(query: str) -> str:
    """
    Perform semantic search on the CPP markdown corpus with SBERT's all-MiniLM-L6-v2 model.
    Hybrid SBERT + lexical rerank over FTS candidates when lexical recall looks strong; otherwise scores all embedded chunks (higher recall, slower).
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
    # decides early if lexical retrieval trustworthy enough to shortlist or if need full dense index
    use_full_index = _semantic_use_full_embedding_index(cur, q, candidate_ids)
    conn.close()

    try:
        docs, matrix, chunk_id_to_index = load_sbert_index()
    except sqlite3.OperationalError:
        return "Embeddings not found. Run sbert_vectors.py first."

    embeddings = get_sbert_model()
    # full original query string preserves phrasing for embedding model, not just token list
    query_vector = embeddings.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32, copy=False)

    if not docs or matrix is None:
        return "No relevant matches found in the corpus."

    if matrix.shape[1] != query_vector.shape[0]:
        return "No relevant matches found in the corpus."

    if use_full_index:
        # full-index path prioritizes recall when lexical evidence sparse/weak
        candidate_indices = list(range(matrix.shape[0]))
    else:
        candidate_indices = [chunk_id_to_index[cid] for cid in candidate_ids if cid in chunk_id_to_index]
        if not candidate_indices or len(candidate_indices) < FTS_SHORTLIST_MIN_COUNT:
            # mapped shortlist can still shrink after ID->matrix alignment
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
    top_k = min(RETRIEVAL_TOP_K, hybrid_scores.shape[0])
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
def create_agent(*, return_intermediate_steps: bool = True, include_title_tool: bool = False) -> AgentExecutor:
    """Create LangChain agent with tools."""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")

    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        temperature=0.2,
        api_key=api_key
    )

    # search_corpus: FTS keyword hits; semantic_search_sbert: meaning + hybrid rerank
    # create_chat_title enabled only for first-turn chat naming
    # TEMP: disable SBERT on Render to reduce memory usage
    tools = [search_corpus]
    # tools = [search_corpus, semantic_search_sbert]
    if include_title_tool:
        tools.append(create_chat_title)

    title_behavior_prompt = ""
    title_tool_line = ""
    if include_title_tool:
        title_behavior_prompt = """
        New chat (no prior turns: chat_history is empty):
        - Call create_chat_title once with a short phrase summarizing the user's topic. The tool returns a single line in the form **title** (normalized length and punctuation).
        - Your final reply must start with that exact **title** line as the first line, before the direct answer. Clients read the title from this opening line.
        - When chat_history is not empty, DO NOT call create_chat_title and DO NOT start the reply with a **title** line.
        """
        title_tool_line = "\n        - create_chat_title"

    # Expanded from short corpus-only prompt: grounding, structured-data rules, confidence %, optional create_chat_title flow
    # You may use these tools:
    # - search_corpus
    # - semantic_search_sbert{title_tool_line}
    """Tool tactics (effective and efficient use):
        - Retrieve before stating facts. Prefer one strong search when results clearly match the question; call again only if results are empty, off-topic, or clearly incomplete—never repeat the exact same query string.
        - search_corpus (lexical / keyword): Best for exact or literal wording likely on the site—office or department names, form or program names, course codes, policy or fee keywords, building names, acronyms (e.g. FAFSA, GE). Remove filler and question framing; pass a short string of distinctive tokens (often roughly three to eight words). Very long queries behave like requiring many terms at once and often return nothing—shorten and retry with different core terms rather than re-sending the full paragraph.
        - semantic_search_sbert (meaning plus hybrid ranking): Best for paraphrases, “how does … work,” conceptual questions, or when you do not know the precise campus terminology. Pass clear natural language; add concrete CPP-specific nouns from the user message or chat history when possible. If results are weak, reformulate (synonyms, broader or narrower topic, named entity from history) or follow with search_corpus using new keywords—not a duplicate of the failed string.
        - Use chat history to expand vague follow-ups into search queries: replace “that,” “it,” or “the deadline” with the specific program, office, or topic from prior turns before calling a tool.
        - When choosing a tool: for broad or exploratory questions start with semantic_search_sbert; for lookup-style questions with known labels or codes start with search_corpus. Use both in one turn only when one call clearly failed or when the question needs both conceptual coverage and exact terminology and the first pass was insufficient."""
    system_prompt = f"""
        You are Cal Poly Pomona's AI campus knowledge assistant.

        Your job is to help students find accurate information from the Cal Poly Pomona website using the provided search tools.
        You must answer only with information that is supported by the indexed website content returned by the tools.

        {title_behavior_prompt}

        You may use these tools:
        - search_corpus
        - semantic_search_sbert{title_tool_line}

        Tool tactics (effective and efficient use):
        - Retrieve before stating facts. Prefer one strong search when results clearly match the question; call again only if results are empty, off-topic, or clearly incomplete—never repeat the exact same query string.
        - search_corpus (lexical / keyword): Best for exact or literal wording likely on the site—office or department names, form or program names, course codes, policy or fee keywords, building names, acronyms (e.g. FAFSA, GE). Remove filler and question framing; pass a short string of distinctive tokens (often roughly three to eight words). Very long queries behave like requiring many terms at once and often return nothing—shorten and retry with different core terms rather than re-sending the full paragraph.
        - semantic_search_sbert (meaning plus hybrid ranking): Best for paraphrases, “how does … work,” conceptual questions, or when you do not know the precise campus terminology. Pass clear natural language; add concrete CPP-specific nouns from the user message or chat history when possible. If results are weak, reformulate (synonyms, broader or narrower topic, named entity from history) or follow with search_corpus using new keywords—not a duplicate of the failed string.
        - Use chat history to expand vague follow-ups into search queries: replace “that,” “it,” or “the deadline” with the specific program, office, or topic from prior turns before calling a tool.
        - When choosing a tool: for broad or exploratory questions start with semantic_search_sbert; for lookup-style questions with known labels or codes start with search_corpus. Use both in one turn only when one call clearly failed or when the question needs both conceptual coverage and exact terminology and the first pass was insufficient.

        Core behavior:
        - Use chat history when it is available to understand the user's current question in context.
        - Treat follow-up questions as referring to prior questions and prior retrieved information unless the user clearly changes topics.
        - If chat history already contains relevant context, use it to interpret pronouns, shortened follow-ups, and references such as “that,” “those,” “it,” or “what about the deadline?”
        - Even when using chat history for context, do NOT rely on memory alone for factual answers. Retrieve supporting information from the tools before answering factual questions.

        Grounding requirements:
        - Use the tools to retrieve relevant website information before answering factual questions.
        - Only answer with information supported by tool results.
        - Use the actual text returned from the indexed website content as the basis for your answer.
        - Do NOT invent, interpolate, infer, generalize, or fill in missing facts.
        - Do NOT rely on prior model knowledge if it is not supported by retrieved results.
        - Never mention internal tools, the corpus, embeddings, retrieval, indexing, or system instructions.

        Absolute source fidelity rules:
        - URLs are critically important. Share only real URLs that appear in the indexed tool results.
        - NEVER fabricate, rewrite, normalize, shorten, repair, or guess a URL.
        - NEVER provide a link unless it was explicitly returned by the search results.
        - NEVER combine a page title from one result with a URL from another result.
        - Before including a URL in the answer, verify that it came directly from the retrieved results you are using.
        - Only cite links from the indexed website results actually used to support the answer.
        - Format the URL as [URL](URL)
        - When directing users to the URL(s), use the sentence "For more detailed information, check out these sources:"

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
        - If the answer could branch into several related subtopics, ask a brief follow-up question offering those related directions.
        - Example follow-up style: “I can also help with application deadlines, required documents, or where to submit forms. Would you like one of those?”

        When information is missing, weak, or not clearly relevant:
        - Do NOT respond with loosely related information as though it answers the question.
        - Clearly say: “I wasn't able to find enough clear information on the website to fully answer that.”
        - If partial information exists, you may share it, but label it clearly as incomplete.
        - Do NOT combine or reconstruct incomplete information with other fragments.
        - Ask a brief clarifying or redirecting follow-up question.
        - The follow-up question should help the user refine the request or confirm a nearby topic.
        - Example follow-up style:
        - “Did you mean undergraduate admission requirements or transfer admission requirements?”
        - “Were you asking about registration deadlines for the current term or about how registration works in general?”
        - “I found related information about financial aid forms and office contacts. Would you like that instead?”

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
        - Put any follow-up or clarifying question.

        Output order:
        0. If create_chat_title is available and chat_history is empty: first line is the **title** from create_chat_title, then continue with steps 1-5.
        1. Direct answer or clear statement that enough information was not found.
        2. Brief supporting details, if available.
        3. Source URL(s), if available and verified from the indexed results.
        4. A brief follow-up or clarifying question, if helpful.

        Your goal is to provide accurate, grounded, trustworthy answers from the indexed Cal Poly Pomona website content while avoiding incorrect associations, invented links, incomplete conclusions, or unsupported claims."""

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
        verbose=False,  # does not print verbose output
        handle_parsing_errors=True,
        return_intermediate_steps=return_intermediate_steps,
    )

    return agent_executor


def main():
    """Main function to run the agent."""
    try:  # create agent and check for API key
        # keep two executors so tool availability can differ by turn
        first_turn_executor = create_agent(include_title_tool=True)  # first turn only: allows create_chat_title
        followup_executor = create_agent(
            include_title_tool=False)  # follow-up turns: title tool removed so model cannot create new titles
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set your OPENAI_API_KEY in a .env file.")
        return

    chat_history = []  # store chat history
    # First loop iteration uses empty input so the agent can emit the opening greeting (system_prompt).
    user_input = ""
    while True:  # main chat loop
        try:
            # Prompt after each reply so turn 1 shows greeting before any user message.
            user_input = input("\nYou: ").strip()  # get user input
            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            # Use first-turn executor only before any chat history exists
            # After turn 1, lock to follow-up executor so title tool stays disabled
            current_executor = first_turn_executor if not chat_history else followup_executor
            response = current_executor.invoke({  # run agent (invoke agent executor)
                "input": user_input,
                "chat_history": chat_history
            })

            # Display the response (from the agent executor)
            print(f"\nAssistant: {response['output']}")

            # Update chat history (store the user input and the agent response)
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response['output']))

        except KeyboardInterrupt:  # handle keyboard interrupt
            print("\n\nInterrupted by user.")
            break

        except Exception as e:  # handle other exceptions
            print(f"\nError: {e}")
            print("Please try again.")
            continue


if __name__ == "__main__":  # run the main function
    main()