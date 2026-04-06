# file to create chat agent to help students navigate website

import os
import time
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv() # load environment variables from .env file

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "campus_kb.db"
CORPUS_DIR = BASE_DIR / "itc2026_ai_corpus"  # raw .md files for brute-force search


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

    match_query = " ".join(q.split())
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
        cur.execute(sql, (match_query,))
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        fallback_terms = [w for w in q.split() if w.strip()]
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

        # Find the first matching line for an excerpt
        excerpt = ""
        for line in text.splitlines():
            if any(term in line.lower() for term in terms):
                excerpt = line.strip()[:500]
                break

        matches.append((md_path.name, excerpt, match_count))

    elapsed = time.perf_counter() - start

    # Sort by most matches first, take top 5
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

# BENCHMARK — run both methods and print timing comparison
def benchmark_search(query: str) -> str:
    print("\n" + "=" * 60)
    print(f"  BENCHMARK: '{query}'")
    print("=" * 60)

    fts_results,   fts_time   = _fts_search(query)
    brute_results, brute_time = _brute_force_search(query)

    speedup = brute_time / fts_time if fts_time > 0 else float("inf")

    print(f"  SQLite FTS5  : {fts_time * 1000:.2f} ms")
    print(f"  Brute-force  : {brute_time * 1000:.2f} ms")
    print(f"  FTS is {speedup:.1f}x faster")
    print("=" * 60 + "\n")

    return fts_results  # agent uses the better result


@tool
def search_corpus(query: str) -> str:
    """
    Search the CPP campus knowledge base for information relevant to the query.
    Runs both SQLite FTS and brute-force search, prints a timing comparison,
    and returns the FTS results for the agent to use.
    """
    return benchmark_search(query)


def create_agent() -> AgentExecutor:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=api_key
    )

    tools = [search_corpus]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant for Cal Poly Pomona students.
        Use the search_corpus tool to find facts in the scraped website corpus.
        Base answers only on tool results; if searches find nothing relevant, say the answer cannot be found in the corpus.
        Keep a conversational tone and cite source URLs from the tool output when you use them."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
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

            # Run the agent (invoke the agent executor)
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            # Display the response (from the agent executor)
            print(f"\nAssistant: {response['output']}")

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