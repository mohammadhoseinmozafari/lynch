from __future__ import annotations
from enum import Enum
from .base import BaseArtifactPointer
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


class ModelArtifactType (Enum) :
    """
    Supported artifact serialization formats and model types.

    This enum defines the recognized artifact types/categories for ML models
    and data objects.

    Attributes:
        PICKLE: Python pickle serialization format.
        JOBLIB: Scikit-learn's joblib format (optimized for large NumPy arrays).
        ONNX: Open Neural Network Exchange format.
        TORCHSCRIPT: PyTorch TorchScript format for deployment.
        PT: PyTorch model format (.pt or .pth files).
        TF_SAVEDMODEL: TensorFlow SavedModel format.
        H5: HDF5 format (commonly used for Keras models).
        CUSTOM: User-defined custom artifact type.
    """
    PICKLE = 'pickle'
    JOBLIB = 'joblib'
    ONNX = 'onnx'
    TORCHSCRIPT = 'torchscript'
    PT = 'pt'
    TF_SAVEDMODEL = 'tf_savedmodel'
    H5 = 'h5'
    CUSTOM = 'custom'
