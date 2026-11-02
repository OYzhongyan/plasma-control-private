"""Durable result cache and OS-level single-worker locking."""
from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import os


@contextmanager
def instance_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / "worker.lock").open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0: handle.write(b"0"); handle.flush()
            handle.seek(0)
            try: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc: raise RuntimeError("Another worker is already running in this data directory") from exc
        else:
            import fcntl
            try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc: raise RuntimeError("Another worker is already running in this data directory") from exc
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == "nt":
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class Store:
    def __init__(self, directory, fingerprint):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "tasks.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, task TEXT NOT NULL, result TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT,
                reported INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        old = self.db.execute("SELECT value FROM settings WHERE key='fingerprint'").fetchone()
        if old and old[0] != fingerprint:
            self.db.close()
            raise ValueError("Data directory fingerprint differs (target/rules/metric/platform/problem changed); use a new data directory")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO settings VALUES ('fingerprint', ?)", (fingerprint,))

    def __enter__(self): return self
    def __exit__(self, *args): self.db.close()

    def register(self, task):
        # Store only necessary submission fields, never API response credentials.
        clean = {k: task[k] for k in ("id", "problem_id", "team_name", "source", "file_url", "status") if k in task}
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO tasks(id,task) VALUES (?,?)", (task["id"], json.dumps(clean)))

    def get(self, sid):
        row = self.db.execute("SELECT * FROM tasks WHERE id=?", (sid,)).fetchone()
        if row is None: return None
        result = dict(row)
        result["task"] = json.loads(result["task"])
        result["result"] = json.loads(result["result"]) if result["result"] else None
        return result

    def attempt(self, sid, error):
        with self.db:
            self.db.execute("UPDATE tasks SET attempts=attempts+1,last_error=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (error, sid))

    def save_result(self, sid, result):
        payload = {**result, "submission_id": sid}
        with self.db:
            self.db.execute("UPDATE tasks SET result=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (json.dumps(payload, ensure_ascii=False, allow_nan=False), sid))

    def note_error(self, sid, message):
        with self.db:
            self.db.execute("UPDATE tasks SET last_error=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (message, sid))

    def mark_reported(self, sid):
        with self.db:
            self.db.execute("UPDATE tasks SET reported=1,updated_at=CURRENT_TIMESTAMP WHERE id=?", (sid,))

    def unreported(self):
        rows = self.db.execute("SELECT id FROM tasks WHERE reported=0 AND result IS NOT NULL ORDER BY rowid").fetchall()
        return [self.get(row[0]) for row in rows]

    def outstanding_count(self):
        return self.db.execute("SELECT COUNT(*) FROM tasks WHERE reported=0").fetchone()[0]
