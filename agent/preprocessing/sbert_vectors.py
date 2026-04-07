"""
sbert_vectors.py

Create vector embeddings for each chunk in SQLite database using SBERT.

What it does:
  1. Reads each chunk from SQLite database
  2. Creates vector embedding for each chunk
  3. Inserts vector embeddings into SQLite database

"""

from __future__ import annotations

import json
import sqlite3
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer

DEFAULT_DB_PATH = "campus_kb.db"
DEFAULT_BATCH_SIZE = 32

CREATE_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS sbert_embeddings (
    chunk_id      INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    embedding     TEXT NOT NULL,
    updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

def init_embeddings_table(conn: sqlite3.Connection) -> None:
    """Create embeddings storage table if missing."""
    cur = conn.cursor()
    cur.executescript(CREATE_EMBEDDINGS_TABLE)
    conn.commit()


def get_chunks_to_embed(conn: sqlite3.Connection, *, limit: int | None = None, only_missing: bool = True) -> list[tuple[int, str]]:
    """
    Return [(chunk_id, chunk_text), ...] for chunks needing embeddings.
    By default, skips rows already present in sbert_embeddings.
    """
    sql = """
        SELECT c.id, c.chunk_text
        FROM chunks c
    """
    if only_missing:
        sql += """
            LEFT JOIN sbert_embeddings e ON e.chunk_id = c.id
            WHERE e.chunk_id IS NULL
        """
    sql += "\nORDER BY c.id"
    if limit and limit > 0:
        sql += f"\nLIMIT {int(limit)}"

    cur = conn.cursor()
    cur.execute(sql)
    return cur.fetchall()


def upsert_embeddings(conn: sqlite3.Connection, rows: list[tuple[int, str]], vectors: list[list[float]]) -> int:
    """Insert/update embeddings in chunk_embeddings keyed by chunk_id."""
    if not rows:
        return 0
    if len(rows) != len(vectors):
        raise ValueError("Row count and vector count do not match.")

    payload = []
    for (chunk_id, _), vector in zip(rows, vectors):
        # store vectors as JSON text so retrieval code can decode to float arrays
        payload.append(
            (
                chunk_id,
                json.dumps(vector),
            )
        )

    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO sbert_embeddings (chunk_id, embedding)
        VALUES (?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            embedding = excluded.embedding
        """,
        payload,
    )
    conn.commit()
    return len(payload)


def batched(seq: list[tuple[int, str]], batch_size: int) -> list[list[tuple[int, str]]]:
    """Yield list batches from a list."""
    return [seq[i : i + batch_size] for i in range(0, len(seq), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/store embeddings for chunks in campus_kb.db.")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite DB")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of chunks to embed")
    parser.add_argument("--update-existing", action="store_true", help="Also regenerate and update embeddings that already exist")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    init_embeddings_table(conn) # creates embedding table only if not already present

    # only generate embeddings for chunks without stored vector
    only_missing = not args.update_existing
    limit = args.limit if args.limit > 0 else None
    rows = get_chunks_to_embed(conn, limit=limit, only_missing=only_missing)
    if not rows:
        print("No chunks require embedding. Nothing to do.")
        conn.close()
        return

    embeddings = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings.max_seq_length = 512
    print(f"Preparing embeddings for {len(rows)} chunks")
    print(f"Model      : all-MiniLM-L6-v2")
    print(f"Batch size : {args.batch_size}")
    print(f"Database   : {db_path.resolve()}\n")

    total_written = 0
    # batch requests to avoid oversized embedding calls and improve throughput
    batches = batched(rows, max(1, args.batch_size))
    for idx, batch_rows in enumerate(batches, start=1):
        texts = [(text or "") for _, text in batch_rows]
        vectors = embeddings.encode(
            texts,
            batch_size=max(1, args.batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()
        written = upsert_embeddings(conn, batch_rows, vectors)
        total_written += written
        print(f"  [{idx:>4}/{len(batches)}] wrote {written:>3} embeddings (total: {total_written})")

    conn.close()
    print(f"\nDone.\n  Embeddings written: {total_written}\n  Database          : {db_path.resolve()}")


if __name__ == "__main__":
    main()