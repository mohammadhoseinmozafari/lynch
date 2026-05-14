from enum import Enum
class DataType (str, Enum) :
    FLOAT64 = 'float64'
    INT64 = 'int64'
    BOOL = 'bool'
    DATETIME64 = 'datetime64'
    TEXT = 'text'