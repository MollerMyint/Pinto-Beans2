# Pinto-Beans2 — CPP AI Knowledge Agent (ITC 2026)

## Goal

Build an AI knowledge agent that answers questions about the Cal Poly Pomona website using tool calling (function calling). The model searches a local knowledge base and must answer from retrieved content, not from guesses. When nothing relevant is found, the agent should say so clearly instead of inventing facts.

## Setup 

---

## 1. Create & Activate Virtual Environment
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```
---

## 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Set Up Environment Variables
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_key_here
```

---

## 4. Preprocess & Index the Corpus
Run these from the root directory with the virtual environment active.
This only needs to be done once (or again if the corpus changes).

```bash
# Clean the raw markdown files (overwrites in place)
python agent/preprocessing/clean_cpp_markdown.py --input-dir ./agent/itc2026_ai_corpus --output-dir ./agent/itc2026_ai_corpus

# Build the SQLite search index
python agent/preprocessing/build_index.py --corpus-dir ./agent/itc2026_ai_corpus --index ./agent/itc2026_ai_corpus/index.json --db ./agent/campus_kb.db

# start creating table of vector embeddings for each chunk (use SBERT instead of the OpenAI model)
# if your local machine can handle a larger batch size, you can include that size in the arguments (ie: --batch-size 256)
python agent/preprocessing/sbert_vectors.py --db agent/campus_kb.db
```
---

## 5.1. Run the Agent on the terminal

Run these from the root directory with the virtual environment active.

```bash
python agent/agent.py
```
**OR if you want to see the difference in retrieval time between generic search and indexing:**
```bash
python agent/agent_test.py
```

## 5.2. Run the Agent with UI

Run this from the root directory with the virtual environment active.

```bash
python -m app.app
```

## 5.3. Run the Agent with Discord

Run this from the root directory with the virtual environment active.

```bash
 python -m discordAgent.discord_bot
```
