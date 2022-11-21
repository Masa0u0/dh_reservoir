import numpy as np

from _ext_lsm import LiquidStateMachine as _LiquidStateMachine
from _ext_lsm import NeuronParams as _NeuronParams
from _ext_lsm import SynapseParams as _SynapseParams


class NeuronParams(_NeuronParams):
    """
    Attributes
    ----------
    tau_decay: float, default 0.02[s]
    c: float, default 7.42[-]
    """

    def __init__(self) -> None:
        super().__init__()


class SynapseParams(_SynapseParams):
    """
    Attributes
    ----------
    u0_mean: float, default 0.5[s]
    u0_sd: float, default 0.25[s]
    tau_d_mean: float, default 0.11[s]
    tau_d_sd: float, default 0.055[s]
    tau_f_mean: float, default 0.005[s]
    tau_f_sd: float, default 0.0025[s]
    c: float, default 7.42[-]
    """

    def __init__(self) -> None:
        super().__init__()


class LiquidStateMachine(_LiquidStateMachine):

    def __init__(
        self,
        pos_input: list,
        pos_wta: list,
        neuron_params: NeuronParams = None,
        synapse_params: SynapseParams = None,
        wta_neurons_range: tuple = (2, 10),
        connectivity: float = 0.5,
        lambda_input: float = 10.,
        lambda_rsrvr: float = 2.,
        stochastic: bool = True,
        max_fire_rate: float = 100.,
        seed: int = None,
    ) -> None:
        if neuron_params is None:
            neuron_params = NeuronParams()
        if synapse_params is None:
            synapse_params = SynapseParams()
        if seed is None:
            seed = -1
        super().__init__(
            pos_input,
            pos_wta,
            neuron_params,
            synapse_params,
            wta_neurons_range,
            connectivity,
            lambda_input,
            lambda_rsrvr,
            stochastic,
            max_fire_rate,
            seed,
        )

    def reset(self) -> None:
        super().reset()

    def step(self, input_spikes: np.ndarray, dt: float) -> np.ndarray:
        rsrvr_traces = np.array(super().step(input_spikes, dt))
        return rsrvr_traces

    @property
    def num_input_neurons(self) -> int:
        return super().get_num_input_neurons()

    @property
    def num_rsrvr_neurons(self) -> int:
        return super().get_num_rsrvr_neurons()

    @property
    def input_connectivity(self) -> float:
        return super().input_connectivity()

    @property
    def rsrvr_connectivity(self) -> float:
        return super().rsrvr_connectivity()
