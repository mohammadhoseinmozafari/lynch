from __future__ import annotations
from typing import  List
from pydantic import BaseModel, ConfigDict
from case.domain.value_objects.chunk_pointer import ChunkPointer

class ArtifactManifest(BaseModel) :
    """
    A manifest describing how a large artifact is split across multiple chunks.

    This model provides assembly instructions for reconstructing an artifact
    that has been divided into smaller chunks for storage or transfer.

    Attributes:
        chunks: List of chunk pointers describing each piece of the artifact.
        assembly_instruction: String specifying how chunks should be combined
            (e.g., order, concatenation method, or custom logic).
        total_size_bytes: Total size of the artifact when all chunks are assembled.
        total_checksum: Combined checksum for verifying the integrity of the
            fully assembled artifact.
    """
    chunks : List[ChunkPointer]
    assembly_instruction : str
    total_size_bytes : int
    total_checksum : str
    model_config = ConfigDict(frozen=True)
