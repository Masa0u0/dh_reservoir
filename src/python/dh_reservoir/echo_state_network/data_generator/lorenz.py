import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

from .base import DataGenerator


class LorenzAttractor(DataGenerator):

    def __init__(self, a: float, b: float, c: float) -> None:
        self._a = a
        self._b = b
        self._c = c

        self._x = np.random.uniform(-1., 1.)
        self._y = np.random.uniform(-1., 1.)
        self._z = np.random.uniform(-1., 1.)

    def step(self, dt: float) -> Tuple[float, float, float]:
        dx = self._a * (self._y - self._x) * dt
        dy = (self._x * (self._b - self._z) - self._y) * dt
        dz = (self._x * self._y - self._c * self._z) * dt

        self._x += dx
        self._y += dy
        self._z += dz

        return self._x, self._y, self._z

    def generate_data(self, label: float, T: float, dt: float) -> Tuple[NDArray, NDArray]:
        assert -1. <= label <= 1.
        assert 0. < dt < T

        data_length = int(T / dt)
        input = np.full((data_length,), label)
        target = np.empty((data_length, 3))

        # 時系列生成
        for k in range(0, data_length):
            xk, yk, zk = self.step(dt)
            target[k, 0] = xk
            target[k, 1] = yk
            target[k, 2] = zk

        return input, target
