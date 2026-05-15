from core.case.domain.value_objects.base_artifact import BaseArtifactPointer
from core.case.domain.enums import ModelArtifactType


class ModelArtifactPointer(BaseArtifactPointer) : 
    """
    A pointer to an artifact stored in a backend storage system.

    This model contains metadata and location information for retrieving
    a machine learning artifact (model, data, etc.) from various storage backends.

    Attributes:
        storage_backend: The storage system where the artifact is located.
        bucket: Optional bucket or container name in the storage backend.
        key: Unique identifier or path for the artifact in the storage backend.
        checksum: Hash value for verifying artifact integrity.
        size_bytes: Size of the artifact in bytes.
        framework: ML framework used to create the artifact (e.g., 'sklearn', 'tensorflow').
        artifact_type: Type/category of the artifact (e.g., PICKLE, ONNX).
        manifest: Optional manifest containing chunk information for large artifacts.
        created_at: Timestamp when the artifact pointer was created.
        tags: Optional dictionary of key-value tags for categorizing the artifact.
    
    """
    framework : str
    artifact_type : ModelArtifactType