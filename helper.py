import os
import pickle

from models import TensorFiles, Embeddings


def loadTensors(file_path: TensorFiles) -> Embeddings:
    """
    This function loads tensors from different types of file
    
    :param file_path: ...
    :type file_path: TensorFiles
    """
    file_path = TensorFiles.file_path
    _, ext = os.path.split(file_path)
    if ext == ".pt":
        ...
    elif ext == ".pkl":
        ...
    ...