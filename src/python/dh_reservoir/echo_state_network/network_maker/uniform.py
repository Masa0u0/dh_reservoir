import numpy as np
import numpy.random as rnd
import numpy.linalg as LA
import networkx as nx
from typing import Tuple

from ..layer import Input, Output, Feedback, Reservoir
from .util import edges_from_weight_matrix


def uniform(
    N_u: int,
    N_y: int,
    N_x: int,
    input_scale: float = 1.,
    fb_scale: float = 0.,
    tau: float = 0.,
    indegree: float = 4.,
    rho: float = 0.95,
) -> Tuple[Input, Output, Feedback, Reservoir]:

    assert N_u > 0 and N_y > 0 and N_x > 0
    assert input_scale > 0.
    assert fb_scale >= 0.
    assert tau >= 0.
    assert indegree > 0.
    assert rho >= 0.

    # Input
    W_in = rnd.uniform(-input_scale, input_scale, (N_x, N_u))
    input_ = Input(W_in)

    # Output
    W_out = rnd.normal(size=(N_y, N_x))
    output = Output(W_out)

    # Feedback
    if fb_scale == 0.:
        feedback = None
    else:
        W_fb = np.random.uniform(-fb_scale, fb_scale, (N_x, N_y))
        feedback = Feedback(W_fb)

    # Reservoir
    # グラフ作成
    total_connections = int(N_x * indegree)
    G = nx.gnm_random_graph(N_x, total_connections)
    W_rsrvr = np.array(nx.to_numpy_matrix(G))

    # 非ゼロ要素を一様分布に従う乱数として生成
    W_rsrvr *= np.random.uniform(-1., 1., (N_x, N_x))

    # スペクトル半径の計算
    eig, _ = LA.eig(W_rsrvr)
    sp_radius = np.max(np.abs(eig))

    # 指定のスペクトル半径rhoに合わせてスケーリング
    W_rsrvr *= rho / sp_radius

    # 非ゼロ要素のみ抽出
    edges = edges_from_weight_matrix(W_rsrvr)

    rsrvr = Reservoir(N_x, tau, edges)

    return input_, output, feedback, rsrvr
