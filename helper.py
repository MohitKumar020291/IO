import os
import pickle
import hashlib

import torch

from models import TensorFiles, Embeddings


def loadTensors(file_path: TensorFiles) -> Embeddings:
    """
    This function loads tensors from different types of file
    
    :param file_path: ...
    :type file_path: TensorFiles
    """
    if not os.path.exists(file_path):
        raise ValueError(f"{file_path} does not exists")
    file_path = TensorFiles.file_path
    _, ext = os.path.split(file_path)
    if ext == ".pt":
        tensors = torch.load(file_path)
        return Embeddings(embeddings=tensors)


def generateFilename(
    base_name: str = None, 
    string: str = None, 
    run_num: int = 1,
    ext: str = None
) -> str:
    if ext and not isinstance(ext, str):
        raise ValueError(f"extension should be of type str, got {type(ext)}")
    if base_name:
        base_name, ext = os.path.splitext(base_name)
    if ext == "" or ext == None: #pain point
        ext = ".pt"
    if not base_name:
        hash_filename = string or "filename"
        sha256_hash = hashlib.sha256()
        sha256_hash.update(hash_filename.encode())
        hex_digest = sha256_hash.hexdigest()
        if os.path.exists(hex_digest + ext):
            return generateFilename(base_name=hex_digest, ext=ext)
        else:
            return hex_digest + ext
    else:
        if os.path.exists(base_name + ext):
            base_name = base_name + str(run_num)
            if os.path.exists(base_name + ext):
                return generateFilename(base_name=base_name, run_num=run_num+1, ext=ext)
            else:
                return base_name + ext
        else:
            return base_name + ext


if __name__ == "__main__":
    print(generateFilename())