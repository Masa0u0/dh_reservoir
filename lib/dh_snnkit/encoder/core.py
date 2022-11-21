import pickle
import json
import numpy as np
from typing import Tuple

from _encoder import PoissonEncoder as _PoissonEncoder
from _encoder import PopSANEncoder as _PopSANEncoder
from _encoder import LIFEncoder as _LIFEncoder


class PoissonEncoder(_PoissonEncoder):

    def __init__(self, obs_dim: int, max_freq: float = 30.) -> None:
        super().__init__(obs_dim, max_freq)

    def encode(self, obs: np.ndarray, dt: float) -> np.ndarray:
        return np.array(super().encode(obs, dt))


class PopSANEncoder(_PopSANEncoder):

    def __init__(
        self,
        obs_dim: int,
        pop_dim: int = 10,
        spike_ts: int = 5,
        std: float = 0.1,
        v_reset: float = 0.,
        v_th: float = 0.999,
        mean_range: tuple = (-1., 1.),
    ) -> None:
        super().__init__(obs_dim, pop_dim, spike_ts, std, v_reset, v_th, mean_range)

    def encode(self, obs: np.ndarray) -> np.ndarray:
        return np.array(super().encode(obs))


class LIFEncoder(_LIFEncoder):

    def __init__(
        self,
        obs_dim: int = None,
        pop_dim: int = 10,
        mean_range: Tuple[float, float] = (-1., 1.),
        std: float = 0.1,
        v_rest: float = 13.5,
        v_reset: float = 13.5,
        v_th: float = 15.,
        tau_m: float = 0.03,
        amp: float = 8.,
        load_path: str = None,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim: int, default None
            観測の次元
        pop_dim: int, default 10
            観測1つあたりのエンコードニューロンの数
        mean_range: Tuple[float, float], default (-1., 1.)
            観測の値域
        std: float, default 0.1
            エンコードニューロンのガウス分布の標準偏差
        v_rest: float, default 13.5
            静止膜電位[mV]
        v_reset: float, default 13.5
            発火後のリセット電位[mV]
        v_th: float, default 15.
            発火の閾値[mV]
        tau_m: float, default 0.03
            膜電位の時定数[s]
        amp: float, default 8.
            エンコーダニューロンの活性度にかかる係数
        load_path: str, default None
            pickleまたはjsonからパラメータを読み込む場合のパス

        Returns
        ----------
        None

        None
        ----------
        [Maass+, 2002]のv_restとr * i_backを合わせてv_restとした。
        """
        if load_path is None:
            assert obs_dim is not None

            self.obs_dim = obs_dim
            self.pop_dim = pop_dim
            self.num_pop_neurons = obs_dim * pop_dim
            self.mean_range = mean_range
            self.std = std
            self.v_rest = v_rest
            self.v_reset = v_reset
            self.v_th = v_th
            self.tau_m = tau_m
            self.amp = amp

        else:
            if load_path.endswith(".pkl"):
                with open(load_path, "rb") as f:
                    params = pickle.load(f)
            elif load_path.endswith(".json"):
                with open(load_path, "r") as f:
                    params = json.load(f)
            else:
                raise AttributeError(f'Invalid file type "{load_path}"')

            self.obs_dim = params["obs_dim"]
            self.pop_dim = params["pop_dim"]
            self.num_pop_neurons = params["num_pop_neurons"]
            self.mean_range = params["mean_range"]
            self.std = params["std"]
            self.v_rest = params["v_rest"]
            self.v_reset = params["v_reset"]
            self.v_th = params["v_th"]
            self.tau_m = params["tau_m"]
            self.amp = params["amp"]

        super().__init__(
            self.obs_dim,
            self.pop_dim,
            self.mean_range,
            self.std,
            self.v_rest,
            self.v_reset,
            self.v_th,
            self.tau_m,
            self.amp,
        )
        self.reset()

    def encode(self, obs: np.ndarray, dt: float) -> np.ndarray:
        """
        アナログデータをスパイクにエンコードする

        Parameters
        ----------
        obs: np.ndarray
            アナログ入力(なるべく[-1, 1]の範囲内にあるのが望ましい)
        dt: float
            前回のエンコーディングからの経過時間

        Returns
        ----------
        spikes: np.ndarray
            その時点のスパイク(0, 1)を記録した配列
        """
        assert obs.shape == (self.obs_dim,)
        return np.array(super().encode(obs, dt))

    def reset(self) -> None:
        """
        内部変数を初期化する

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # init_v = np.random.uniform(self.v_reset, self.v_th, (self.num_pop_neurons,))
        init_v = np.full((self.num_pop_neurons,), self.v_reset)
        super().reset(init_v)

    def save(self, save_path: str) -> None:
        """
        LIFエンコーダのパラメータをpickleまたはjson形式で保存する

        Parameters
        ----------
        save_path: str
            保存先のパス

        Returns
        ----------
        None
        """
        params = dict()
        params["obs_dim"] = self.obs_dim
        params["pop_dim"] = self.pop_dim
        params["num_pop_neurons"] = self.num_pop_neurons
        params["mean_range"] = self.mean_range
        params["std"] = self.std
        params["v_rest"] = self.v_rest
        params["v_reset"] = self.v_reset
        params["v_th"] = self.v_th
        params["tau_m"] = self.tau_m
        params["amp"] = self.amp

        if save_path.endswith(".pkl"):
            with open(save_path, "wb") as f:
                pickle.dump(params, f)
        elif save_path.endswith(".json"):
            with open(save_path, "w") as f:
                json.dump(params, f, indent=4)
        else:
            raise AttributeError(f'Invalid file type "{save_path}"')
