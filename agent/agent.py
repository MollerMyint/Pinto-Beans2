# file to create chat agent to help students navigate website

import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from flask import Flask, request, render_template
import warnings

load_dotenv() # load environment variables from .env file
warnings.filterwarnings('ignore')

# ============================================================================
# LANGCHAIN TOOLS (TBD: IMPLEMENT THESE)
# ============================================================================
# These functions are decorated with @tool to make them available to the
# LangChain agent. The agent can call these tools to perform specific tasks.

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "campus_kb.db"


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

# Flask constructor
app = Flask(__name__)

@tool
@app.route('/ask', methods=['POST'])
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

    conn.close()
    return format_results(rows)

def create_agent() -> AgentExecutor:
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

    tools = [search_corpus]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant for Cal Poly Pomona students.
        Use the search_corpus tool to find facts in the scraped website corpus (markdown under test_corpus).
        Base answers only on tool results; if searches find nothing relevant, say the answer cannot be found in the corpus.
        Keep a conversational tone and cite source URLs from the tool output when you use them."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Create the agent (uses the tools and prompt)
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Create executor (runs the agent)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False, # does not print verbose output
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