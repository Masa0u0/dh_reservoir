import numpy as np
import numpy.random as rnd
from numpy.typing import NDArray
from typing import Dict, List

from dh_function.random import sc_normal, sc_lognormal, sc_gamma


def conmat2pairs(conmat: NDArray) -> NDArray:
    assert conmat.ndim == 2
    pairs = []
    for i in range(0, conmat.shape[0]):
        for j in range(0, conmat.shape[1]):
            if conmat[i][j]:
                pairs.append([i, j])
    return np.array(pairs, dtype=int)


def set_syn_type(pairs, is_exc1: NDArray, is_exc2: NDArray) -> List[str]:
    n_syn = pairs.shape[0]
    syn_type = [None] * n_syn
    for i, (pre, post) in enumerate(pairs):
        if is_exc1[pre] and is_exc2[post]:
            syn_type[i] = "ee"   # 興奮性→興奮性
        elif is_exc1[pre] and (not is_exc2[post]):
            syn_type[i] = "ei"   # 興奮性→抑制性
        elif (not is_exc1[pre]) and is_exc2[post]:
            syn_type[i] = "ie"   # 抑制性→興奮性
        else:
            syn_type[i] = "ii"   # 抑制性→抑制性
    return syn_type


def sample_syn_params(
    syn_type: List[str],
    mean_dict: Dict[str, float],
    sd_dict: Dict[str, float],
    distribution: str,
    std_coef: float,
) -> NDArray:
    num_syn = len(syn_type)
    res = np.empty((num_syn,))
    for i in range(0, num_syn):
        mean = mean_dict[syn_type[i]]
        sd = sd_dict[syn_type[i]]
        lb = mean - std_coef * sd
        ub = mean + std_coef * sd
        if distribution == "uniform":
            res[i] = rnd.uniform(lb, ub)
        elif distribution == "sc_normal":
            res[i] = sc_normal(mean, sd, (lb, ub))
        elif distribution == "sc_lognormal":
            res[i] = sc_lognormal(mean, sd, (lb, ub))
        elif distribution == "sc_gamma":
            res[i] = sc_gamma(mean, sd, (lb, ub))
        else:
            raise ValueError(f'unknown distribution "{distribution}"')
    return res
