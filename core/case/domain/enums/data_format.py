from enum import Enum

class DataFormat(str, Enum) :
    PARQUET = 'parquet'
    CSV = 'csv'
    TF_RECORD = 'tf_record'
    NUMPY = 'numpy'
    ## arrow (?)
