
from enum import Enum

class DataCompression(str, Enum) :
    GZIP = 'gzip'
    SNAPPY = 'snappy'
    ZSTD = 'zstd'
    LZ4 = 'lz4'
