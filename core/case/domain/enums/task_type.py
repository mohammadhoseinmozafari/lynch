from enum import Enum

class TaskType(str,Enum) :
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    RANKING = 'ranking'
    CLUSTERING = 'clustering'
    GENERATIVE = 'generative'