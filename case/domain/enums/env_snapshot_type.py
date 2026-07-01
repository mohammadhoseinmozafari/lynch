from enum import Enum

class EnvSnapshotType (str,Enum) :
    CONDA_LOCK = 'conda-lock'
    PIP_LOCK =  'pip-lock'
    #DOCKER_DIGEST = 'docker-digest' to be added
