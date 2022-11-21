import pickle
import numpy as np
import numpy.linalg as LA
import numpy.random as rnd
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from matplotlib import patches
from rich import print
from typing import Tuple, List

from dh_algorithm.tree import UnionFind

from _liquid_state_machine import LiquidStateMachine as _LiquidStateMachine
from _liquid_state_machine import LSMParam as _LSMParam


class LSMParam:

    def __init__(
        self,
        num_input: int,
        num_rsrvr: int,
        num_syn_input: int,
        num_syn_rsrvr: int,
        tau_m: float,
        v_th: float,
        v_rest: float,
        v_reset: float,
        i_back: float,
        i_noise_scale: float,
        r: float,
        is_exc_input: List[bool],
        is_exc_rsrvr: List[bool],
        pairs_input: List[Tuple[int, int]],
        pairs_rsrvr: List[Tuple[int, int]],
        t_ref: NDArray,
        tau_decay: NDArray,
        tau_syn_input: NDArray,
        tau_syn_rsrvr: NDArray,
        tau_d: NDArray,
        tau_f: NDArray,
        u0: NDArray,
        a_input: NDArray,
        a_rsrvr: NDArray,
        delay_input: NDArray,
        delay_rsrvr: NDArray,
    ) -> None:
        self.num_input = num_input
        self.num_rsrvr = num_rsrvr
        self.num_syn_input = num_syn_input
        self.num_syn_rsrvr = num_syn_rsrvr
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.i_back = i_back
        self.i_noise_scale = i_noise_scale
        self.r = r
        self.is_exc_input = is_exc_input
        self.is_exc_rsrvr = is_exc_rsrvr
        self.pairs_input = pairs_input
        self.pairs_rsrvr = pairs_rsrvr
        self.t_ref = t_ref
        self.tau_decay = tau_decay
        self.tau_syn_input = tau_syn_input
        self.tau_syn_rsrvr = tau_syn_rsrvr
        self.tau_d = tau_d
        self.tau_f = tau_f
        self.u0 = u0
        self.a_input = a_input
        self.a_rsrvr = a_rsrvr
        self.delay_input = delay_input
        self.delay_rsrvr = delay_rsrvr

    def __call__(self):
        return _LSMParam(
            self.num_input,
            self.num_rsrvr,
            self.num_syn_input,
            self.num_syn_rsrvr,
            self.tau_m,
            self.v_th,
            self.v_rest,
            self.v_reset,
            self.i_back,
            self.i_noise_scale,
            self.r,
            self.is_exc_input,
            self.is_exc_rsrvr,
            self.pairs_input,
            self.pairs_rsrvr,
            self.t_ref,
            self.tau_decay,
            self.tau_syn_input,
            self.tau_syn_rsrvr,
            self.tau_d,
            self.tau_f,
            self.u0,
            self.a_input,
            self.a_rsrvr,
            self.delay_input,
            self.delay_rsrvr,
        )


class LiquidStateMachine(_LiquidStateMachine):

    def __init__(
        self,
        param: LSMParam = None,
        load_path: str = None,
        seed: int = None,
    ) -> None:
        """
        Parameters
        ----------
        param: LSMParam, default None
        load_path: str, default None
        seed: str, default None

        Returns
        ----------
        None

        Note
        ----------
        param, load_pathのいずれか一方のみを指定する
        """
        assert (param and (load_path is None)) or ((param is None) and load_path)
        assert seed is None or seed >= 0

        if param:
            self._param = param
            self._connection_rate = self._calc_connection_rate()
            self._eig = self._calc_eig()
            # self._num_unstable_eig = sum(abs(self._eig) > 1.)
        else:
            assert load_path.endswith(".pkl")
            with open(load_path, "rb") as f:
                attributes = pickle.load(f)
            self._param = attributes["param"]
            self._connection_rate = attributes["connection_rate"]
            self._eig = attributes["eig"]
            # self._num_unstable_eig = attributes["num_unstable_eig"]

        # グラフが連結かどうかの確認
        if self._connection_rate < 1.:
            print(
                "[yellow]Warning: LSM graph is disconnected.[/yellow]"
                f'[yellow] The connection rate is {self._connection_rate:.3f}.[/yellow]'
            )

        # 再帰結合行列の固有値を確認
        # if self._num_unstable_eig > 0:
        #     print(
        #         f'[yellow]Warning: {self._num_unstable_eig} out of {self.num_rsrvr}[/yellow]'
        #         "[yellow] eigenvalues are out of unit circle.[/yellow]"
        #     )

        super().__init__(self._param(), -1 if (seed is None) else seed)
        self.reset()

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
        init_v = rnd.uniform(self._param.v_reset, self._param.v_th, (self._param.num_rsrvr,))
        # init_v = np.full((self.num_rsrvr,), self._param.v_reset)
        super().reset(init_v)

    def step(self, input_spikes: NDArray, dt: float) -> NDArray:
        """
        dtだけシミュレートする

        Parameters
        ----------
        input_spikes: NDArray

        Returns
        ----------
        rsrvr_traces: NDArray
        """
        assert input_spikes.shape == (self._param.num_input,)
        rsrvr_traces = np.array(super().step(input_spikes, dt))
        return rsrvr_traces

    def start_log(self) -> None:
        return super().start_log()

    def _calc_connection_rate(self):
        uf = UnionFind(n=self._param.num_rsrvr)
        for pre, post in self._param.pairs_rsrvr:
            uf.unite(pre, post)

        num_total_pairs = num_connected_pairs = 0
        for i in range(0, self._param.num_rsrvr):
            for j in range(0, i):
                num_total_pairs += 1
                if uf.same(i, j):
                    num_connected_pairs += 1

        connection_rate = num_connected_pairs / num_total_pairs
        return connection_rate

    def _make_synapse_strength_matrix(self) -> NDArray:
        w = self._param.tau_syn_rsrvr * self._param.a_rsrvr
        w_mat = np.zeros((self._param.num_rsrvr, self._param.num_rsrvr))
        for i, (pre, post) in enumerate(self._param.pairs_rsrvr):
            w_mat[pre, post] = w[i]
        return w_mat

    def _calc_eig(self) -> NDArray:
        w_mat = self._make_synapse_strength_matrix()
        eig, _ = LA.eig(w_mat)
        return eig

    def plot_synapse_strength(self, figsize: Tuple[float, float] = (6., 6.)) -> None:
        w_mat = self._make_synapse_strength_matrix()
        tmp = abs(w_mat)
        tmp = tmp / np.max(tmp) * 255
        plt.figure(figsize=figsize)
        plt.title("Synapse Strength")
        plt.imshow(tmp, cmap="gray")

    def plot_eig(self, figsize: Tuple[float, float] = (6., 6.)) -> None:
        """
        再帰結合行列の固有値を描画する

        Parameters
        ----------
        figsize: Tuple[float, float], default (6., 6.)

        Returns
        ----------
        None
        """
        x = self._eig.real
        y = self._eig.imag
        r = abs(self._eig)
        # stable_idx = np.where(r <= 1.)
        # unstable_idx = np.where(r > 1.)
        print(
            "Absolute value of the eigenvalues:\n"
            f'  min   : {np.min(r):3f}\n'
            f'  max   : {np.max(r):3f}\n'
            f'  mean  : {np.mean(r):.3f}\n'
            f'  median: {np.median(r):.3f}\n'
        )

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        # ax.set_title("Eigenvalue Spectrum")
        lim = max(2., np.max(r) + 0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # ax.scatter(x[stable_idx], y[stable_idx], s=1., color="b")
        # ax.scatter(x[unstable_idx], y[unstable_idx], s=1., color="r")
        ax.scatter(x, y, s=1., color="k")
        for r in range(1, 30):
            ax.add_patch(patches.Circle(xy=(0., 0.), radius=r, ec="k", fill=False))

    def plot_spike_log(self, dt: float, figsize: Tuple[float, float] = (8., 6.)) -> None:
        """
        リザバー層の全ニューロンのスパイクのログを表示する

        Parameters
        ----------
        figsize: Tuple[float, float], default (8., 6.)

        Returns
        ----------
        None
        """
        num_spikes = [0] * self._param.num_rsrvr
        exc_data = [[-1, -1]]   # 全く発火していない場合のためにダミーを入れておく
        inh_data = [[-1, -1]]
        for step, idxes in enumerate(self.spike_log):
            t = dt * step
            for idx in idxes[:-1]:   # 末尾の要素はダミー
                num_spikes[idx] += 1
                if self._param.is_exc_rsrvr[idx]:
                    exc_data.append([t, idx])
                else:
                    inh_data.append([t, idx])
        exc_data = np.array(exc_data)
        inh_data = np.array(inh_data)

        print(f'mean fire rate: {np.mean(num_spikes) / (len(self.spike_log) * dt):.3f}Hz')

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        # ax.set_title("Spike Timing")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Neuron Index")
        ax.set_xlim(0, dt * len(self.spike_log))
        ax.set_ylim(0, self._param.num_rsrvr)
        ax.scatter(exc_data[:, 0], exc_data[:, 1], s=0.1, color="r")
        ax.scatter(inh_data[:, 0], inh_data[:, 1], s=0.1, color="b")

    def save(self, path: str):
        """
        LSMのパラメータをpickle形式で保存する

        Parameters
        ----------
        path: str

        Returns
        ----------
        None
        """
        assert path.endswith(".pkl")

        attributes = dict()
        attributes["param"] = self._param
        attributes["connection_rate"] = self._connection_rate
        attributes["eig"] = self._eig
        # attributes["num_unstable_eig"] = self._num_unstable_eig

        with open(path, "wb") as f:
            pickle.dump(attributes, f)

    @property
    def num_input(self) -> int:
        return self._param.num_input

    @property
    def num_rsrvr(self) -> int:
        return self._param.num_rsrvr

    @property
    def tau_decay(self) -> NDArray:
        return self._param.tau_decay

    @property
    def input_connectivity(self) -> float:
        return self._param.num_syn_input / (self._param.num_input * self._param.num_rsrvr)

    @property
    def rsrvr_connectivity(self) -> float:
        return self._param.num_syn_rsrvr / (self._param.num_rsrvr**2)

    @property
    def rsrvr_trace(self) -> NDArray:
        return np.array(super().get_trace())

    @property
    def spike_log(self) -> list:
        return super().get_spike_log()
