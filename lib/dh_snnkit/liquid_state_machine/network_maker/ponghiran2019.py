# https://arxiv.org/abs/1906.01695 の動的シナプスver

import numpy as np
import numpy.random as rnd
from typing import Tuple, Dict

from ..core import LSMParam
from .util import conmat2pairs, set_syn_type, sample_syn_params


def ponghiran2019(
    num_input: int,
    num_rsrvr: int,
    k: float = 3.,
    c: float = 1.5,
    exc_rate: float = 0.8,
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
    num_rsrvr: int,
        レザバー層のニューロン数
    k: float, default 3.
        入力層からレザバー層(興奮性)への平均入次数
    c: float, default 1.5
        レザバー層(興奮性)からレザバー層(抑制性)への入次数(逆も同じ)
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
    assert num_input > 0 and num_rsrvr > 0
    assert len(tau_decay_range) == 2 and tau_decay_range[0] <= tau_decay_range[1]
    assert v_rest < v_reset < v_th

    num_exc = int(num_rsrvr * exc_rate)
    num_inh = num_rsrvr - num_exc
    is_exc_input = np.ones((num_input,), dtype=int)
    is_exc_rsrvr = np.r_[np.ones((num_exc,)), np.zeros((num_inh,))].astype(int)

    # 結合の作成
    conmat_input = rnd.rand(num_input, num_exc) < k / num_input
    pairs_input = conmat2pairs(conmat_input)

    conmat_ei = rnd.rand(num_exc, num_inh) < c / num_exc
    pairs_ei = conmat2pairs(conmat_ei)
    pairs_ei[:, 1] += num_exc

    conmat_ie = rnd.rand(num_inh, num_exc) < c / num_inh
    pairs_ie = conmat2pairs(conmat_ie)
    pairs_ie[:, 0] += num_exc

    conmat_ee = conmat_ei @ conmat_ie   # 興奮性のみのループを排除
    for i in range(0, num_exc):
        conmat_ee[i, i] = False         # 自己ループを排除
    pairs_ee = conmat2pairs(conmat_ee)

    conmat_ii = conmat_ie @ conmat_ei   # 抑制性のみのループを排除
    for i in range(0, num_inh):
        conmat_ii[i, i] = False         # 自己ループを排除
    pairs_ii = conmat2pairs(conmat_ii)

    pairs_rsrvr = np.r_[pairs_ei, pairs_ie, pairs_ee, pairs_ii]

    # シナプスタイプを設定
    num_syn_input = pairs_input.shape[0]
    num_syn_rsrvr = pairs_rsrvr.shape[0]
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
