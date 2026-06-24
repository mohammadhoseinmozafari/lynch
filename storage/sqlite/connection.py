import sqlite3
from contextlib import contextmanager
from typing import Generator


class SQLiteDB:
    def __init__(self, path: str = "holmes.db"):
        self.path = path

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()