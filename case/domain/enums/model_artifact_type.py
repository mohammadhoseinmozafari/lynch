from enum import Enum


class ModelArtifactType (str, Enum) :
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
