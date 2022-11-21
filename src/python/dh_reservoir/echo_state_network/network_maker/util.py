from numpy.typing import NDArray
from typing import List

from _reservoir import Edge


def edges_from_weight_matrix(weight_matrix: NDArray) -> List[Edge]:
    """
    結合行列から辺のリストを作成する．

    Parameters
    ----------
    weight_matrix : NDArray[ndim=2]
        結合行列

    Returns
    -------
    edges : List[Edge]
        辺のリスト
    """
    assert weight_matrix.ndim == 2

    edges = []
    for i in range(0, weight_matrix.shape[0]):
        for j in range(0, weight_matrix.shape[1]):
            if weight_matrix[i, j] != 0:
                edge = Edge(i, j, weight_matrix[i, j])
                edges.append(edge)

    return edges
