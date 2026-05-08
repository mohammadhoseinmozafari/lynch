from typing import Optional
from pydantic import BaseModel


class CodeSnapshot (BaseModel) :
    git_repository : str
    git_commit_hash : str
    entry_point : str
    git_branch : Optional[str]
    uncommmitted_diff : Optional[str]
    training_command : Optional[str]
    code_snapshot_hash : Optional[str]