# Pinto-Beans2 — CPP AI Knowledge Agent (ITC 2026)

## Goal

Build an AI knowledge agent that answers questions about the Cal Poly Pomona website using tool calling (function calling). The model searches a local knowledge base and must answer from retrieved content, not from guesses. When nothing relevant is found, the agent should say so clearly instead of inventing facts.

## Setup (brief)

1. Create a `.env` with `OPENAI_API_KEY=...`.
2. Install dependencies (e.g. `pip install -r requirements.txt`).
3. From `Server/`, run: `python agent.py`.
