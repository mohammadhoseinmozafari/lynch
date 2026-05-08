from __future__ import annotations
from enum import Enum
from typing import Optional
from .base import BaseArtifactPointer



class DataArtifactPointer (BaseArtifactPointer) :
    data_format : DataFormat
    compression : Optional[Compression] = None
    record_count : Optional[int] = None
    column_count : Optional [int] = None


class DataFormat(Enum) :
    PARQUET = 'parquet'
    CSV = 'csv'
    TF_RECORD = 'tf_record'
    NUMPY = 'numpy'
    ## arrow (?)

class Compression(Enum) :
    GZIP = 'gzip'
    SNAPPY = 'snappy'
    ZSTD = 'zstd'
    LZ4 = 'lz4'
