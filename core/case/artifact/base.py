from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field

class StorageBackend (str,Enum) :
    """
    Supported storage backends for artifact storage and retrieval.

    This enum defines the available storage systems where artifacts can be
    persistently stored.

    Attributes:
        LOCAL: Local filesystem storage.
        S3: Amazon Simple Storage Service.
        GCS: Google Cloud Storage.
        AZURE_BLOB: Microsoft Azure Blob Storage.
        MINIO: MinIO object storage (S3-compatible).
    """
    LOCAL = 'local'
    S3 = 's3'
    GCS = 'gcs'
    AZURE_BLOB = 'azure_blob'
    MINIO = 'minio'

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

class BaseArtifactPointer(BaseModel) : 
    """
    A base pointer class to an artifact stored in a backend storage system.

    This model contains metadata and location information for retrieving
    an artifact (model, data, etc.) from various storage backends.

    Attributes:
        storage_backend: The storage system where the artifact is located.
        bucket: Optional bucket or container name in the storage backend.
        key: Unique identifier or path for the artifact in the storage backend.
        checksum: Hash value for verifying artifact integrity.
        size_bytes: Size of the artifact in bytes.
        artifact_type: Type/category of the artifact (e.g., PICKLE, ONNX).
        manifest: Optional manifest containing chunk information for large artifacts.
        created_at: Timestamp when the artifact pointer was created.
        tags: Optional dictionary of key-value tags for categorizing the artifact.
    
    """
    storage_backend : StorageBackend
    bucket : Optional[str] = None
    key : str
    checksum : str ## needs custom validation
    size_bytes : int = Field(ge=1)
    manifest : Optional[ArtifactManifest] = None
    created_at : Optional[datetime] = None
    tags : Optional[dict[str, str]] = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)





