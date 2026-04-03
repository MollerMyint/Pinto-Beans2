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
python Server/preprocessing/clean_cpp_markdown.py --input-dir ./Server/itc2026_ai_corpus --output-dir ./Server/itc2026_ai_corpus

# Build the SQLite search index
python Server/preprocessing/build_index.py --corpus-dir ./Server/itc2026_ai_corpus --index ./Server/itc2026_ai_corpus/index.json --db ./Server/campus_kb.db
```
---

## 5. Run the Agent
```bash
python Server/agent.py
```
