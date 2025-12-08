from typing import Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_pop_idxs(arr: np.ndarray, threshold: Union[int, float] = 1):
    token_diff_thresh = 5
    pop_indexes = []
    prev_preserved = 0
    for idx, (i, j) in enumerate(zip(arr[:-1], arr[1:])):
        if j - i <= token_diff_thresh and j - arr[prev_preserved] <= token_diff_thresh:
            pop_indexes.append(idx+1)
        else:
            prev_preserved = idx+1
    return pop_indexes

def sort_argsort(*args: list[np.ndarray], sort_idxs: np.ndarray) -> np.ndarray:
    returns = []
    for arr in args:
        if not isinstance(arr, np.ndarray):
            raise ValueError("arr must be an numpy array")
        arr = arr[sort_idxs]
        returns.append(arr)
    return tuple(returns)

def do_pop_idxs(*args: list[np.ndarray], pop_idxs: np.ndarray) -> np.ndarray:
    returns = []
    k = 0
    for arr in args:
        if not isinstance(arr, np.ndarray):
            raise ValueError("arr must be an numpy array")
        for pop_idx in pop_idxs:
            arr = np.delete(arr, pop_idx - k)
            k += 1
        returns.append(arr)
    return tuple(returns)

def tfIdf(documents: list[str], query: str):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    for i, score in enumerate(similarities):
        print(f"Document {i}: similarity = {score:.4f}")

    best_index = similarities.argmax()
    return best_index
