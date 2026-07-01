from enum import Enum

class DataFeatureType (str, Enum) :
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknow"
    ## text_length
    ## geospatial
