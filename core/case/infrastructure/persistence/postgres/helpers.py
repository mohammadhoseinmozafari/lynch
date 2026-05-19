import json
from datetime import datetime
from uuid import UUID

def serialize_uuid(u: UUID) -> str:
    return str(u)

def deserialize_uuid(s: str) -> UUID:
    return UUID(s)

def serialize_datetime(dt: datetime) -> str:
    return dt.isoformat()

def deserialize_datetime(s: str) -> datetime:
    return datetime.fromisoformat(s)

def serialize_json(obj) -> str:
    return json.dumps(obj, default=str)

def deserialize_json(s: str):
    return json.loads(s) if s else None