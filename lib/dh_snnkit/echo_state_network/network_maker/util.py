import numpy as np


def conmat2pairs(conmat: np.ndarray) -> np.ndarray:
    assert conmat.ndim == 2

    pairs = []
    weights = []

    for i in range(0, conmat.shape[0]):
        for j in range(0, conmat.shape[1]):
            if conmat[i, j] != 0:
                pairs.append([i, j])
                weights.append(conmat[i, j])

    return np.array(pairs, dtype=int), np.array(weights, dtype=float)
