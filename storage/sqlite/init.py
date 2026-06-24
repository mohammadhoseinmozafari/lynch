from storage.sqlite.schema import SCHEMA
from storage.sqlite.pragmas import PRAGMAS


class SQLiteInitializer:
    def __init__(self, db):
        self.db = db

    def init(self):
        with self.db.connect() as conn:
            for p in PRAGMAS:
                conn.execute(p)

            for stmt in SCHEMA:
                conn.execute(stmt)