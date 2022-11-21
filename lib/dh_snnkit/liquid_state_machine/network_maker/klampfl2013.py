# https://www.jneurosci.org/content/jneuro/33/28/11515.full.pdf

import numpy as np
import numpy.random as rnd
import numpy.linalg as LA
from typing import Tuple, Dict
from rich import print

from ..core import LSMParam
from .util import set_syn_type, sample_syn_params


def klampfl2013(
    num_input: int,
    num_wta: int,
    wta_neurons_range: Tuple[int, int] = (2, 6),
    k: float = 3.,
    lam: float = 0.01,
    tau_m: float = 0.03,
    tau_decay_range: Tuple[float, float] = (0.03, 0.03),
    t_ref_dict: Dict[str, float] = {"exc": 3e-3, "inh": 2e-3},
    v_th: float = 15.,
    v_rest: float = 0.,
    v_reset: float = 13.5,
    i_back: float = 13.5,
    i_noise_scale: float = 8.,
    r: float = 1.,
    tau_syn_mean_dict: Dict[str, float] = {"ee": 3e-3, "ei": 3e-3, "ie": 6e-3, "ii": 6e-3},
    tau_syn_sd_dict: Dict[str, float] = {"ee": 0., "ei": 0., "ie": 0., "ii": 0.},
    tau_d_mean_dict: Dict[str, float] = {"ee": 1.1, "ei": 0.125, "ie": 0.7, "ii": 0.144},
    tau_d_sd_dict: Dict[str, float] = {"ee": 0.55, "ei": 0.0625, "ie": 0.35, "ii": 0.072},
    tau_f_mean_dict: Dict[str, float] = {"ee": 0.05, "ei": 1.2, "ie": 0.02, "ii": 0.06},
    tau_f_sd_dict: Dict[str, float] = {"ee": 0.025, "ei": 0.6, "ie": 0.01, "ii": 0.03},
    u0_mean_dict: Dict[str, float] = {"ee": 0.5, "ei": 0.05, "ie": 0.25, "ii": 0.32},
    u0_sd_dict: Dict[str, float] = {"ee": 0.25, "ei": 0.025, "ie": 0.125, "ii": 0.16},
    a_in_mean_dict: Dict[str, float] = {"ee": 18., "ei": 9., "ie": -18., "ii": -9.},
    a_in_sd_dict: Dict[str, float] = {"ee": 18., "ei": 9., "ie": 18., "ii": 9.},
    a_re_mean_dict: Dict[str, float] = {"ee": 30., "ei": 60., "ie": -19., "ii": -19.},
    a_re_sd_dict: Dict[str, float] = {"ee": 30., "ei": 60., "ie": 19., "ii": 19.},
    delay_dict: Dict[str, float] = {"ee": 0., "ei": 0., "ie": 0., "ii": 0.},
    tau_distribution: str = "sc_normal",
    tau_std_coef: float = 1.,
    u0_distribution: str = "sc_normal",
    u0_std_coef: float = 1.,
    a_distribution: str = "sc_gamma",
    a_std_coef: float = 10.,
) -> LSMParam:
    """
    Parameters
    ----------
    num_input: int,
        入力層のニューロン数
    num_wta: int,
        WTA回路の数
    wta_neurons_range: Tuple[int, int], default (2, 6)
        1つのWTA回路を構成する興奮性ニューロン数の範囲(両端を含む)
    k: float, default 3.
        入力層からレザバー層(興奮性)への平均入次数
    lam: float, default 0.01
        レザバー層の結合率の分布形状を決める定数
    tau_m: float, default 0.03
        膜電位の時定数[s]
    tau_decay_range: Tuple[float, float], default (0.03, 0.03)
        発火トレースの時定数[s]
    t_ref_dict: Dict[str, float], default {"exc": 3e-3, "inh": 2e-3}
        不応期[s]
    v_th: float, default 15.
        発火の閾値[mV]
    v_rest: float, default 0.
        静止膜電位[mV]
    v_reset: float, default 13.5
        発火後のリセット電位[mV]
    i_back: float, default 13.5
        環境からの定常入力電流[nA]
    i_noise_scale: float, default 8.
        ノイズ電流(所謂ゆらぎ)のスケール[nA]
    r: float, default 1.
        膜抵抗[MΩ]
    tau_syn_mean_dict: Dict[str, float], default {"ee": 3e-3, "ei": 3e-3, "ie": 6e-3, "ii": 6e-3}
        シナプス後電流の減衰の時定数の平均[s]
    tau_syn_sd_dict: Dict[str, float], default {"ee": 0., "ei": 0., "ie": 0., "ii": 0.}
        シナプス後電流の減衰の時定数の標準偏差[s]
    tau_d_mean_dict: Dict[str, float], default {"ee": 1.1, "ei": 0.125, "ie": 0.7, "ii": 0.144}
        内部パラメータXの時定数の平均[s]
    tau_d_sd_dict: Dict[str, float], default {"ee": 0.55, "ei": 0.0625, "ie": 0.35, "ii": 0.072}
        内部パラメータXの時定数の標準偏差[s]
    tau_f_mean_dict: Dict[str, float], default {"ee": 0.05, "ei": 1.2, "ie": 0.02, "ii": 0.06}
        内部パラメータUの時定数の平均[s]
    tau_f_sd_dict: Dict[str, float], default {"ee": 0.025, "ei": 0.6, "ie": 0.01, "ii": 0.03}
        内部パラメータUの時定数の標準偏差[s]
    u0_mean_dict: Dict[str, float], default {"ee": 0.5, "ei": 0.05, "ie": 0.25, "ii": 0.32}
        発火による内部パラメータUの変化量の平均[-]
    u0_sd_dict: Dict[str, float], default {"ee": 0.25, "ei": 0.025, "ie": 0.125, "ii": 0.16}
        発火による内部パラメータUの変化量の標準偏差[-]
    a_in_mean_dict: Dict[str, float], default {"ee": 18., "ei": 9., "ie": 18., "ii": 9.}
        入力層のシナプス強度の平均[nA]
    a_in_sd_dict: Dict[str, float], default {"ee": 18., "ei": 9., "ie": 18., "ii": 9.}
        入力層のシナプス強度の標準偏差[nA]
    a_re_mean_dict: Dict[str, float], default {"ee": 30., "ei": 60., "ie": -19., "ii": -19.}
        リザバー層のシナプス強度の平均[nA]
    a_re_sd_dict: Dict[str, float], default {"ee": 30., "ei": 60., "ie": 19., "ii": 19.}
        リザバー層のシナプス強度の標準偏差[nA]
    delay_dict: Dict[str, float], default {"ee": 0., "ei": 0., "ie": 0., "ii": 0.}
        シナプス後電流の伝播遅延の時定数[s]
    X_distribution: str
        Xをサンプリングする際の確率分布("uniform", "sc_normal", "sc_lognormal", "sc_gamma")
    X_std_coef: float
        XをSoft Clipする際に第coef標準偏差まで採用する

    Returns
    ----------
    param: LSMParam
        LiquidStateMachineのパラメータの構造体
    """
    assert num_input > 0 and num_wta > 0
    assert 0. < lam < 1.
    assert len(wta_neurons_range) == 2 and 0 < wta_neurons_range[0] <= wta_neurons_range[1]
    assert len(tau_decay_range) == 2 and tau_decay_range[0] <= tau_decay_range[1]
    assert v_rest < v_reset < v_th

    # rsrvr→rsrvrの結合
    wta_exc_idxes = [[] for _ in range(num_wta)]
    wta_inh_idxes = []
    is_exc_rsrvr = np.empty((0,), dtype=int)
    num_rsrvr = 0
    for i in range(num_wta):
        num_neuron = rnd.randint(wta_neurons_range[0], wta_neurons_range[1] + 1)
        for _ in range(num_neuron):
            wta_exc_idxes[i].append(num_rsrvr)
            is_exc_rsrvr = np.append(is_exc_rsrvr, 1)
            num_rsrvr += 1
    for _ in range(num_wta):
        wta_inh_idxes.append(num_rsrvr)
        is_exc_rsrvr = np.append(is_exc_rsrvr, 0)
        num_rsrvr += 1

    a = np.power(rnd.uniform(0., 1., (num_wta, 1)), 1. / 3.)
    theta = rnd.uniform(0., 2. * np.pi, (num_wta,))
    phi = rnd.uniform(0., 2. * np.pi, (num_wta,))
    units = np.c_[
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(theta),
        np.cos(theta),
    ]
    pos_wta = a * units

    pairs_rsrvr = np.empty((0, 2), dtype=int)
    # WTA内の結合
    for exc_idxes, inh_idx in zip(wta_exc_idxes, wta_inh_idxes):
        for exc_idx in exc_idxes:
            pairs_rsrvr = np.vstack((pairs_rsrvr, [exc_idx, inh_idx]))
            pairs_rsrvr = np.vstack((pairs_rsrvr, [inh_idx, exc_idx]))
    # WTA間の結合
    for i in range(num_wta):
        for j in range(num_wta):
            if i == j:
                continue
            dist = LA.norm(pos_wta[i, :] - pos_wta[j, :], ord=2)
            prob = lam * np.exp(- lam * dist)
            assert 0. < prob < 1.
            for idx_i in wta_exc_idxes[i]:
                for idx_j in wta_exc_idxes[j]:
                    if rnd.rand() < prob:
                        pairs_rsrvr = np.vstack((pairs_rsrvr, [idx_i, idx_j]))

    # input→rsrvrの結合
    is_exc_input = np.ones((num_input,), dtype=int)
    pairs_input = np.empty((0, 2), dtype=int)
    for input_idx in range(num_input):
        for exc_idxes in wta_exc_idxes:
            for exc_idx in exc_idxes:
                if rnd.rand() < k / num_input:
                    pairs_input = np.vstack((pairs_input, [input_idx, exc_idx]))

    num_syn_input = pairs_input.shape[0]
    num_syn_rsrvr = pairs_rsrvr.shape[0]
    print(f'リザバーニューロン数　: {num_rsrvr}')
    print(f'リザバー層の平均入次数: {num_syn_rsrvr / num_rsrvr:.3f}')

    # シナプスタイプを設定
    syn_type_input = set_syn_type(pairs_input, is_exc_input, is_exc_rsrvr)
    syn_type_rsrvr = set_syn_type(pairs_rsrvr, is_exc_rsrvr, is_exc_rsrvr)

    # ニューロンのパラメータの設定
    tau_decay = rnd.uniform(tau_decay_range[0], tau_decay_range[1], (num_rsrvr,))
    t_ref = np.where(is_exc_rsrvr, t_ref_dict["exc"], t_ref_dict["inh"])

    # シナプスのパラメータの設定
    tau_syn_input = sample_syn_params(
        syn_type_input, tau_syn_mean_dict, tau_syn_sd_dict, tau_distribution, tau_std_coef,
    )
    tau_syn_rsrvr = sample_syn_params(
        syn_type_rsrvr, tau_syn_mean_dict, tau_syn_sd_dict, tau_distribution, tau_std_coef,
    )
    tau_d = sample_syn_params(
        syn_type_rsrvr, tau_d_mean_dict, tau_d_sd_dict, tau_distribution, tau_std_coef,
    )
    tau_f = sample_syn_params(
        syn_type_rsrvr, tau_f_mean_dict, tau_f_sd_dict, tau_distribution, tau_std_coef,
    )
    u0 = sample_syn_params(
        syn_type_rsrvr, u0_mean_dict, u0_sd_dict, u0_distribution, u0_std_coef,
    )
    a_input = sample_syn_params(
        syn_type_input, a_in_mean_dict, a_in_sd_dict, a_distribution, a_std_coef,
    )
    a_rsrvr = sample_syn_params(
        syn_type_rsrvr, a_re_mean_dict, a_re_sd_dict, a_distribution, a_std_coef,
    )
    delay_input = sample_syn_params(
        syn_type_input, delay_dict, {"ee": 0., "ei": 0., "ie": 0., "ii": 0.}, "uniform", 0.,
    )
    delay_rsrvr = sample_syn_params(
        syn_type_rsrvr, delay_dict, {"ee": 0., "ei": 0., "ie": 0., "ii": 0.}, "uniform", 0.,
    )

    # LSMParamオブジェクトの作成
    param = LSMParam(
        num_input=num_input,
        num_rsrvr=num_rsrvr,
        num_syn_input=num_syn_input,
        num_syn_rsrvr=num_syn_rsrvr,
        tau_m=tau_m,
        v_th=v_th,
        v_rest=v_rest,
        v_reset=v_reset,
        i_back=i_back,
        i_noise_scale=i_noise_scale,
        r=r,
        is_exc_input=is_exc_input,
        is_exc_rsrvr=is_exc_rsrvr,
        pairs_input=pairs_input,
        pairs_rsrvr=pairs_rsrvr,
        t_ref=t_ref,
        tau_decay=tau_decay,
        tau_syn_input=tau_syn_input,
        tau_syn_rsrvr=tau_syn_rsrvr,
        tau_d=tau_d,
        tau_f=tau_f,
        u0=u0,
        a_input=a_input,
        a_rsrvr=a_rsrvr,
        delay_input=delay_input,
        delay_rsrvr=delay_rsrvr,
    )

    return param
