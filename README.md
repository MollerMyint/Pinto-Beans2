# Pinto-Beans2 - CPP AI Knowledge Agent (ITC 2026)

This project is a Cal Poly Pomona campus knowledge assistant. It answers questions using retrieved website corpus content (not freeform guessing), and supports both:
- a terminal chatbot flow
- a web app with login/chat history

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

3. **LLM agent layer (in `agent/agent.py`)**
   - Builds a LangChain tools agent with OpenAI (`ChatOpenAI`).
   - The system prompt enforces source-grounded behavior and anti-hallucination rules.
   - Chat history is passed to keep multi-turn context.

4. **Web app layer (in `app/app.py`)**
   - Handles login/signup/session and chat APIs.
   - Persists user chats/messages in MySQL tables (`users`, `chats`, `messages`).
   - Calls the same agent from `agent/agent.py`.

### Retrieval logic

- The agent has two retrieval tools:
  - **`search_corpus(query)`**: lexical (FTS5 / BM25), good for exact terms.
  - **`semantic_search_sbert(query)`**: semantic retrieval with SBERT (`all-MiniLM-L6-v2`) plus lexical overlap reranking.
- Semantic search can switch between:
  - an FTS-shortlisted candidate set (faster), or
  - scoring the full embedding index (higher recall) when lexical evidence is weak.
- Final answer text is generated only after retrieval, and is expected to cite URLs returned by the index.

## Setup Instructions

### 1) Create and activate a virtual environment

From repo root:

On Mac:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 2) Install dependencies

This repo keeps dependencies in `./requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3) Get an OpenAI API key

#### How to get the key
1. Go to [OpenAI Platform](https://platform.openai.com/).
2. Sign in and create/select a project.
3. Open API keys and create a new secret key.
4. Copy it immediately (you will not be able to view it again later).

## Database Usage Guide

### A) Retrieval DB (SQLite) - `agent/campus_kb.db`

**Purpose:** AI search and retrieval over CPP corpus.  

---

### Setup Options

#### Option 1 — Download Prebuilt DB

download the database and place it in the `agent/` folder: [campus_kb.db](https://github.com/MollerMyint/Pinto-Beans2/releases/download/version1.0/campus_kb.db)

So now you should have:
```text
agent/campus_kb.db
```

---

#### Option 2 — Build Locally

Before running the preprocessing scripts, download the corpus and place it in the `agent/` folder : [itc2026_ai_corpus.zip](https://github.com/MollerMyint/Pinto-Beans2/releases/download/version1.0/itc2026_ai_corpus.zip)

Extract it so now you should have:

```text
agent/itc2026_ai_corpus/
```

Then run:

```bash
# Clean and normalize raw markdown files
python agent/preprocessing/clean_cpp_markdown.py \
  --input-dir ./agent/itc2026_ai_corpus \
  --output-dir ./agent/itc2026_ai_corpus
```
```bash
# Build the retrieval database (SQLite)
python agent/preprocessing/build_index.py \
  --corpus-dir ./agent/itc2026_ai_corpus \
  --index ./agent/itc2026_ai_corpus/index.json \
  --db ./agent/campus_kb.db \
  --rebuild
```
```bash
# Generate vector embeddings (SBERT)
python agent/preprocessing/sbert_vectors.py --db ./agent/campus_kb.db
```
---

> `campus_kb.db` and `itc2026_ai_corpus.zip` are not included in the repo due to size.  

---

### B) Application DB (MySQL) - for users/chats/messages

**Purpose:** authentication and chat persistence in the web UI.  

---

### Setup

Use the included [chat_bot.sql](./chat_bot.sql) file to create the required tables.

#### Option 1 — Run via terminal 

```bash
mysql -u <username> -p <database_name> < chat_bot.sql
```

---

#### Option 2 — Use MySQL Workbench

1. Open MySQL Workbench
2. Connect to your database  
3. Open `chat_bot.sql`  
4. Click **Execute (⚡)** to run the script  

---

### Tables Created
- `users` (`user_id`, `username`, `emailaddress`, `password`)
- `chats` (`chat_id`, `user_id`, `title`, `created_at`)
- `messages` (`message_id`, `chat_id`, `question`, `answer`)
  
---

> If these tables are missing, the web app routes will fail.

## Configure environment variables (`.env`)

Create a `.env` file in the root directory and add the following:

```env
# OpenAI API (used by the agent)
OPENAI_API_KEY=your_openai_api_key_here

# MySQL connection (used by app/app.py)
DB_HOSTNAME=your_db_host
DB_USERNAME=your_db_user
DB_PASSWORD=your_db_password
DB_PORT=3306
DB_DATABASE=your_db_name
DB_SSL=/absolute/path/to/ca.pem

# Password hashing salt (used for login/signup)
SALT=your_random_salt_string
```

> This file is required for the app to connect to the database and run properly.

## Running the Project (run all of these commands from the root directory)

### Terminal chatbot

```bash
python agent/agent.py
```

### Retrieval testing script

```bash
python agent/agent_test.py
```

### Web app

```bash
python -m app.app
```

## Discord app

```bash
 python -m discordAgent.discord_bot
```

The default port in this project is `5001`.

## Project Tree

```text
Pinto-Beans2/
├── .gitignore
├── .env                        # (create this yourself - not included)       
├── README.md
├── chat_bot.sql
├── requirements.txt
├── agent/
│   ├── agent.py
    ├── itc2026_ai_corpus/      # (only included if you choose option 2 in database usage guide)
    ├── campus_kb.db            # (generate or add locally - not included)
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
        ├── account.css
    │   ├── auth.css
    │   ├── index.css
    │   ├── BillyBronco.png
    │   └── CPP.png
    └── templates/
        ├── account.html
        ├── index.html
        ├── login.html
        └── signup.html
```
