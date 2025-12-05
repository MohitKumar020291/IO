import torch
import numpy as np

import os
from pydantic import BaseModel, field_validator, ConfigDict


class TensorFiles(BaseModel):
    """
    Purpose of the class is served by check_pt
    """
    file_path: str

    @field_validator('file_path')
    @classmethod
    def check_pt(cls, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension not in [".pt"]:
            raise ValueError("pt_file_path should have extension .pt")
        return file_path

class Prompt(BaseModel):
    prompt: str

class Tokens(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tokens: torch.Tensor

    def __len__(self):
        return self.tokens.size(0)


class Embeddings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    embeddings: np.ndarray
