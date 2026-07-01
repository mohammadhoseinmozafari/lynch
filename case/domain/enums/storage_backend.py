from __future__ import annotations
from enum import Enum

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