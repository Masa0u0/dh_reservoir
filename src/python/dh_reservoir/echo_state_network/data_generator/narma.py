import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

from .base import DataGenerator


class NARMA(DataGenerator):

    def __init__(self, order: int, a1: float, a2: float, a3: float, a4: float) -> None:
        assert order > 0

        self._order = order
        self._a1 = a1
        self._a2 = a2
        self._a3 = a3
        self._a4 = a4

    def generate_data(self, data_length: int) -> Tuple[NDArray, NDArray]:
        assert data_length > self._order

        y = [0.] * self._order
        u = np.random.uniform(0, 0.5, data_length)

        # 時系列生成
        for k in range(self._order, data_length):
            yn_1 = self._a1 * y[k - 1]
            yn_2 = self._a2 * y[k - 1] * np.sum(y[k - self._order:k - 1])
            yn_3 = self._a3 * u[k - self._order] * u[k]
            yn_4 = self._a4
            yn = yn_1 + yn_2 + yn_3 + yn_4
            y.append(yn)

        return u, np.array(y)
