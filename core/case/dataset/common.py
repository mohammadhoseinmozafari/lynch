from enum import Enum
class DataSplitName(str,Enum) :
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

class DataFeatureType (str, Enum) :
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    #DATETIME = "datetime"
    ## text_length
    ## geospatial


class DataType (str, Enum) :
    FLOAT64 = 'float64'
    INT64 = 'int64'
    BOOL = 'bool'
    DATETIME64 = 'datetime64'
    TEXT = 'text'


class TaskType(str,Enum) :
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    RANKING = 'ranking'
    CLUSTERING = 'clustering'
    GENERATIVE = 'generative'