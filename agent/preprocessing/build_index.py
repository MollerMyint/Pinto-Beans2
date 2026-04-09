"""
build_index.py

Build the SQLite full-text-search database used by agent.py.
 
What it does:
  1. Reads index.json and maps source URLs to markdown filenames
  2. Reads each .md file from the corpus directory
  3. Splits each file into headed chunks (H1/H2 boundaries, ~800 chars max)
  4. Inserts documents + chunks into campus_kb.db 
"""
 
from __future__ import annotations
 
import re
import json
import sqlite3
import argparse
from pathlib import Path
 
 
# Config defaults
DEFAULT_CORPUS_DIR = "."          # directory that contains the .md files
DEFAULT_INDEX_JSON = "index.json" # URL -> filename map
DEFAULT_DB_PATH    = "campus_kb.db"
MAX_CHUNK_CHARS    = 800          # soft cap before splitting on blank line
 
# Markdown helpers
HEADING_RE = re.compile(r"^(#{1,2})\s+(.+)", re.MULTILINE)
 
 
def extract_title(md_text: str) -> str | None:
    """Return the first H1 heading text, or None."""
    m = re.search(r"^#\s+(.+)", md_text, re.MULTILINE)
    return m.group(1).strip() if m else None
 
 
def chunk_markdown(md_text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[str | None, str]]:
    """
    Split markdown into (heading, chunk_text) pairs.
        - Walk the file line by line, tracking the current H1/H2 heading.
        - Accumulate lines until we hit another heading OR the chunk exceeds max_chars AND we see a blank line (natural paragraph break).
        - Yield each non-empty chunk with its heading.
    """
    lines = md_text.splitlines()
    chunks: list[tuple[str | None, str]] = []
 
    current_heading: str | None = None
    buffer: list[str] = []
 
    def flush(heading: str | None, buf: list[str]) -> None:
        text = "\n".join(buf).strip()
        if text:
            chunks.append((heading, text))
 
    for line in lines:
        h_match = re.match(r"^(#{1,2})\s+(.+)", line)
        if h_match:
            flush(current_heading, buffer)
            buffer = []
            current_heading = h_match.group(2).strip()
            buffer.append(line)
            continue
 
        buffer.append(line)
 
        # split on blank line if buffer is getting long
        current_text = "\n".join(buffer)
        if len(current_text) >= max_chars and line.strip() == "":
            flush(current_heading, buffer)
            buffer = []
 
    flush(current_heading, buffer)
    return chunks


# Database setup
CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name  TEXT NOT NULL,
    title      TEXT,
    source_url TEXT
);
"""
 
CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    heading     TEXT,
    chunk_text  TEXT NOT NULL
);
"""
 
CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
USING fts5(heading, chunk_text, content='chunks', content_rowid='id');
"""
 
# Keep the FTS index in sync with the chunks table via triggers
TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, heading, chunk_text)
    VALUES (new.id, new.heading, new.chunk_text);
END;
 
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, heading, chunk_text)
    VALUES ('delete', old.id, old.heading, old.chunk_text);
END;
 
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, heading, chunk_text)
    VALUES ('delete', old.id, old.heading, old.chunk_text);
    INSERT INTO chunks_fts(rowid, heading, chunk_text)
    VALUES (new.id, new.heading, new.chunk_text);
END;
"""
 
 
def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(CREATE_DOCUMENTS + CREATE_CHUNKS + CREATE_FTS + TRIGGERS)
    conn.commit()
 
 
def clear_db(conn: sqlite3.Connection) -> None:
    """Drop all rows so we can do a clean rebuild."""
    cur = conn.cursor()
    cur.executescript("""
        DELETE FROM chunks_fts;
        DELETE FROM chunks;
        DELETE FROM documents;
    """)
    conn.commit()
 
# Indexing logic
def index_file(conn: sqlite3.Connection, file_name: str, source_url: str, md_text: str) -> tuple[int, int]:
    """
    Insert one markdown file into the DB.
    Returns (num_chunks_inserted, 0) for progress reporting.
    """
    title = extract_title(md_text)
    chunks = chunk_markdown(md_text)
 
    if not chunks:
        return 0, 0
 
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (file_name, title, source_url) VALUES (?, ?, ?)",
        (file_name, title, source_url),
    )
    doc_id = cur.lastrowid
 
    cur.executemany(
        "INSERT INTO chunks (document_id, heading, chunk_text) VALUES (?, ?, ?)",
        [(doc_id, heading, text) for heading, text in chunks],
    )
    conn.commit()
    return len(chunks), 0
 
# Main
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CPP campus knowledge base SQLite index.")
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR, help="Directory containing .md files")
    parser.add_argument("--index",      default=DEFAULT_INDEX_JSON,  help="Path to index.json")
    parser.add_argument("--db",         default=DEFAULT_DB_PATH,     help="Output SQLite database path")
    parser.add_argument("--rebuild",    action="store_true",         help="Clear existing data before indexing")
    args = parser.parse_args()
 
    corpus_dir = Path(args.corpus_dir)
    index_path = Path(args.index)
    db_path    = Path(args.db)
 
    # Load URL -> filename map
    if not index_path.exists():
        print(f"ERROR: index file not found: {index_path}")
        return
 
    with index_path.open(encoding="utf-8") as f:
        url_to_file: dict[str, str] = json.load(f)
 
    print(f"Loaded {len(url_to_file)} entries from {index_path}")
 
    # Reverse to filename -> url (some filenames may map to multiple URLs; last one wins)
    file_to_url: dict[str, str] = {v: k for k, v in url_to_file.items()}
 
    # Connect / init DB
    conn = sqlite3.connect(db_path)
    init_db(conn)
 
    if args.rebuild:
        print("Clearing existing data for rebuild...")
        clear_db(conn)
 
    # Walk corpus
    total_docs   = 0
    total_chunks = 0
    missing      = 0
    skipped      = 0
 
    md_files = sorted(corpus_dir.glob("*.md"))
    if not md_files:
        # Also try one level deep (cleaned/ subfolder, etc.)
        md_files = sorted(corpus_dir.rglob("*.md"))
 
    print(f"Found {len(md_files)} .md files in '{corpus_dir}'\n")
 
    for md_path in md_files:
        file_name  = md_path.name
        source_url = file_to_url.get(file_name, "")
 
        # Skip files not in index.json unless they're standalone pages
        if not source_url:
            # Still index them with a blank URL so they're searchable
            source_url = ""
 
        try:
            md_text = md_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  SKIP (read error): {file_name} — {e}")
            skipped += 1
            continue
 
        if not md_text.strip():
            skipped += 1
            continue
 
        n_chunks, _ = index_file(conn, file_name, source_url, md_text)
        total_docs   += 1
        total_chunks += n_chunks
 
        status = f"  [{total_docs:>4}] {file_name:<65} {n_chunks:>3} chunks"
        if not source_url:
            status += "  (no URL in index)"
        print(status)
 
    conn.close()
 
    print(f"""
Done.
  Documents indexed : {total_docs}
  Total chunks      : {total_chunks}
  Skipped           : {skipped}
  Database          : {db_path.resolve()}
""")
 
 
if __name__ == "__main__":
    main()