import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple


class NARMA:
    """ NARMA生成モデル """

    def __init__(self, m: int, a1: float, a2: float, a3: float, a4: float) -> None:
        self.m = m
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    def generate_data(self, T_train: int, y_init: List[float]) -> Tuple[NDArray, NDArray]:
        n = self.m
        y = y_init
        u = np.random.uniform(0, 0.5, T_train)

        # 時系列生成
        while n < T_train:
            yn_1 = self.a1 * y[n - 1]
            yn_2 = self.a2 * y[n - 1] * (np.sum(y[n - self.m:n - 1]))
            yn_3 = self.a3 * u[n - self.m] * u[n]
            yn_4 = self.a4
            yn = yn_1 + yn_2 + yn_3 + yn_4
            y.append(yn)
            n += 1

        return u, np.array(y)
