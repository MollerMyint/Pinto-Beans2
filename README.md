# Pinto-Beans2 - CPP AI Knowledge Agent (ITC 2026)

This project is a Cal Poly Pomona campus knowledge assistant. It answers questions using retrieved website corpus content (not freeform guessing), and supports both:
- a terminal chatbot flow
- a Flask web app with login/chat history

## Architecture and Logic

### High-level architecture

1. **Corpus preprocessing and indexing**
   - Raw markdown pages are cleaned by `agent/preprocessing/clean_cpp_markdown.py`.
   - `agent/preprocessing/build_index.py` chunks markdown pages and stores them in SQLite.
   - `agent/preprocessing/sbert_vectors.py` creates SBERT embeddings and stores them in SQLite.

2. **Retrieval layer (SQLite in `agent/campus_kb.db`)**
   - `search_corpus` uses SQLite FTS5 lexical retrieval.
   - `semantic_search_sbert` uses SBERT embeddings, then hybrid-ranks results.
   - Both tools return source-attributed chunks (title, heading, URL, excerpt).

3. **LLM agent layer**
   - `agent/agent.py` builds a LangChain tools agent with OpenAI (`ChatOpenAI`).
   - The system prompt enforces source-grounded behavior and anti-hallucination rules.
   - Chat history is passed to keep multi-turn context.

4. **Web app layer (Flask in `app/app.py`)**
   - Handles login/signup/session and chat APIs.
   - Persists user chats/messages in MySQL tables (`users`, `chats`, `messages`).
   - Calls the same agent from `agent/agent.py`.

### Retrieval logic (important)

- The agent has two retrieval tools:
  - **`search_corpus(query)`**: lexical (FTS5 / BM25), good for exact terms.
  - **`semantic_search_sbert(query)`**: semantic retrieval with SBERT (`all-MiniLM-L6-v2`) plus lexical overlap reranking.
- Semantic search can switch between:
  - an FTS-shortlisted candidate set (faster), or
  - scoring the full embedding index (higher recall) when lexical evidence is weak.
- Final answer text is generated only after retrieval, and is expected to cite URLs returned by the index.

## Current Project Tree

```text
Pinto-Beans2/
├── .gitignore
├── README.md
├── agent/
│   ├── agent.py
│   ├── agent_test.py
│   └── preprocessing/
│       ├── build_index.py
│       ├── clean_cpp_markdown.py
│       ├── openai_vectors.py
│       └── sbert_vectors.py
└── app/
    ├── app.py
    ├── requirements.txt
    ├── static/
    │   ├── auth.css
    │   ├── index.css
    │   ├── BillyBronco.png
    │   └── CPP.png
    └── templates/
        ├── index.html
        ├── login.html
        └── signup.html
```

## Setup Instructions

### 1) Create and activate virtual environment

From repo root:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 2) Install dependencies

This repo keeps dependencies in `app/requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3) Download and place the text corpus

You must download the corpus folder and place it **inside `agent/`** with this exact name:

```text
agent/itc2026_ai_corpus
```

Required: `index.json` must be inside that folder, along with the markdown files.

### 4) Get an OpenAI API key and configure `.env`

#### How to get the key
1. Go to [OpenAI Platform](https://platform.openai.com/).
2. Sign in and create/select a project.
3. Open API keys and create a new secret key.
4. Copy it immediately (you will not be able to view it again later).

#### Add the key to environment variables
Create `.env` in repo root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5) Configure database environment variables (`.env`)

The Flask app also expects MySQL and hashing environment variables:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# MySQL connection (used by app/app.py)
DB_HOSTNAME=your_db_host
DB_USERNAME=your_db_user
DB_PASSWORD=your_db_password
DB_PORT=3306
DB_DATABASE=your_db_name
DB_SSL=/absolute/path/to/ca.pem

# Password hashing salt (used in signup/login)
SALT=your_random_salt_string
```

## Database Usage Guide

This project uses **two databases** for different purposes:

### A) Retrieval DB (SQLite) - `agent/campus_kb.db`

Purpose: AI search and retrieval over CPP corpus.

Built by preprocessing scripts and used by `agent/agent.py`.

Core tables:
- `documents`: page-level metadata (`file_name`, `title`, `source_url`)
- `chunks`: chunked markdown text per document
- `chunks_fts`: FTS5 virtual table for lexical search
- `sbert_embeddings`: vector embedding per chunk

To build/rebuild:

```bash
# Optional clean pass over markdown
python agent/preprocessing/clean_cpp_markdown.py \
  --input-dir ./agent/itc2026_ai_corpus \
  --output-dir ./agent/itc2026_ai_corpus

# Build document/chunk/FTS tables
python agent/preprocessing/build_index.py \
  --corpus-dir ./agent/itc2026_ai_corpus \
  --index ./agent/itc2026_ai_corpus/index.json \
  --db ./agent/campus_kb.db \
  --rebuild

# Generate SBERT embeddings
python agent/preprocessing/sbert_vectors.py --db ./agent/campus_kb.db
```

### B) Application DB (MySQL) - for users/chats/messages

Purpose: authentication and chat persistence in the web UI.

Used in `app/app.py` (via `mysql.connector`).

Referenced tables:
- `users` (contains `user_id`, `username`, `emailaddress`, `password`)
- `chats` (contains `chat_id`, `user_id`, `title`, `created_at`)
- `messages` (contains `message_id`, `chat_id`, `question`, `answer`, `confidence`)

If these tables do not exist in your MySQL database, the web app routes will fail.

## Running the Project

### Terminal chatbot

```bash
python agent/agent.py
```

### Retrieval testing script

```bash
python agent/agent_test.py
```

### Flask web app

```bash
python -m app.app
```

Default Flask port in this project is `5001`.
