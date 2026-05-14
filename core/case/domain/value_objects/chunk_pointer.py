from __future__ import annotations
from pydantic import BaseModel, ConfigDict

class ChunkPointer(BaseModel) :
    """
    A pointer to a single chunk of a multi-part artifact.

    This model contains location and metadata for one piece of a larger artifact
    that has been split into multiple chunks.

    Attributes:
        chunk_index: Zero-based index indicating the chunk's position in the sequence.
        key: Unique identifier or path for this chunk in the storage backend.
        checksum: Hash value for verifying this chunk's integrity.
        size_bytes: Size of this chunk in bytes.
    """
    chunk_index : int
    key : str
    checksum : str
    size_bytes : int

    model_config= ConfigDict(frozen=True)