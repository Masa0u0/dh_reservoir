import numpy as np

from _reservoir import Reservoir as _Reservoir


class Input:

    def __init__(self, W_in: np.ndarray) -> None:
        assert W_in.ndim == 2
        self.N_x = W_in.shape[0]
        self.N_u = W_in.shape[1]
        self.W_in = W_in

    def __call__(self, u: np.ndarray) -> np.ndarray:
        assert u.shape == (self.N_u,)
        return np.dot(self.W_in, u)


class Output:

    def __init__(self, W_out: np.ndarray) -> None:
        assert W_out.ndim == 2
        self.N_y = W_out.shape[0]
        self.N_x = W_out.shape[1]
        self.W_out = W_out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (self.N_x,)
        return np.dot(self.W_out, x)

    def set_weight(self, W_out_opt: np.ndarray) -> None:
        assert W_out_opt.shape == self.W_out.shape
        self.W_out = W_out_opt


class Feedback:

    def __init__(self, W_fb: np.ndarray) -> None:
        assert W_fb.ndim == 2
        self.N_x = W_fb.shape[0]
        self.N_y = W_fb.shape[1]
        self.W_fb = W_fb

    def __call__(self, y: np.ndarray) -> np.ndarray:
        assert y.shape == (self.N_y,)
        return np.dot(self.W_fb, y)


class Reservoir(_Reservoir):

    def __init__(self, N_x: int, tau: float, pairs: np.ndarray, weights: np.ndarray):
        self.N_x = N_x
        self.tau = tau
        self.pairs = pairs
        self.weights = weights
        super().__init__(N_x, tau, pairs, weights)

    def step(self, x_in: np.ndarray, dt: float) -> np.ndarray:
        assert x_in.shape == (self.N_x,)
        assert 0 < dt < self.tau
        return np.array(super().step(x_in, dt))

    def reset(self) -> None:
        super().reset()

    def get_state(self) -> np.ndarray:
        return np.array(super().get_state())

    def get_attributes(self) -> dict:
        return {
            'N_x': self.N_x,
            'tau': self.tau,
            'pairs': self.pairs,
            'weights': self.weights,
        }
