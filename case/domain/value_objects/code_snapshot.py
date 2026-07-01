from typing import Optional
from pydantic import BaseModel, Field


class CodeSnapshot (BaseModel) :
    git_repository : str = Field(min_length=1)
    git_commit_hash : str = Field (pattern='^[a-f0-9]{7,40}$')
    entry_point : str = Field(min_length=1)
    git_branch : Optional[str]
    uncommitted_diff : Optional[str]
    training_command : Optional[str]