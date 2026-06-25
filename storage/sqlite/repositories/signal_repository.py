import json
from datetime import datetime
from typing import Any, List

from core.signal.signal import Signal, SignalEvent
from storage.sqlite.connection import SQLiteDB


class SignalRepository:
	def __init__(self, db: SQLiteDB) -> None:
		self.db = db

	def insert(self, signal: Signal | SignalEvent) -> None:
		self.insert_many([signal])

	def insert_many(self, signals: List[Signal | SignalEvent]) -> None:
		if not signals:
			return

		events = [self._as_event(signal) for signal in signals]

		with self.db.connect() as conn:
			conn.executemany(
				"""
				INSERT OR REPLACE INTO signals
				(
					id,
					created_at,
					category,
					signal_type,
					subject_type,
					subject_name,
					value,
					confidence,
					extractor_id,
					payload
				)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				[
					(
						event.id,
						self._to_timestamp(event.created_at),
						event.category,
						event.signal_type,
						event.subject_type,
						event.subject_name,
						event.value,
						event.confidence,
						event.extractor_id,
						json.dumps(event.payload),
					)
					for event in events
				],
			)

			relation_rows = [
				(event.id, observation_id)
				for event in events
				for observation_id in event.source_observation_ids
			]
			if relation_rows:
				conn.executemany(
					"""
					INSERT OR IGNORE INTO signal_observations
					(signal_id, observation_id)
					VALUES (?, ?)
					""",
					relation_rows,
				)

	def fetch_all(self) -> List[Any]:
		with self.db.connect() as conn:
			return conn.execute("SELECT * FROM signals").fetchall()

	def fetch_by_category(self, category: str) -> List[Any]:
		with self.db.connect() as conn:
			return conn.execute(
				"SELECT * FROM signals WHERE category = ?",
				(category,),
			).fetchall()

	def fetch_by_signal_type(self, signal_type: str) -> List[Any]:
		with self.db.connect() as conn:
			return conn.execute(
				"SELECT * FROM signals WHERE signal_type = ?",
				(signal_type,),
			).fetchall()

	def fetch_observation_ids_for_signal(self, signal_id: str) -> List[str]:
		with self.db.connect() as conn:
			rows = conn.execute(
				"""
				SELECT observation_id
				FROM signal_observations
				WHERE signal_id = ?
				""",
				(signal_id,),
			).fetchall()
			return [row["observation_id"] for row in rows]

	@staticmethod
	def _as_event(signal: Signal | SignalEvent) -> SignalEvent:
		if isinstance(signal, SignalEvent):
			return signal
		return signal.to_event()

	@staticmethod
	def _to_timestamp(value: datetime) -> int:
		return int(value.timestamp())
