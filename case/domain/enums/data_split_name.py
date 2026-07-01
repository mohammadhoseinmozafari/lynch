from enum import Enum

class DataSplitName(str,Enum) :
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'