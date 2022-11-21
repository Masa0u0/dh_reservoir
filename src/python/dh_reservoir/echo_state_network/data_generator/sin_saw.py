import numpy as np

from .base import DataGenerator


class SinSaw(DataGenerator):
    """ 正弦波とのこぎり波の混合波形生成 """

    def __init__(self, freq: float, dt: float):
        self.freq = freq
        self.period = 1. / freq
        self.dt = dt
        self.period_length = int(self.period / self.dt)
        assert self.period_length > 0

    def generate_data(self, label):
        '''
        混合波形及びラベルの出力
        :param label: 0または1を要素に持つリスト
        :return: u: 混合波形
        :return: d: 2次元ラベル（正弦波[1,0], のこぎり波[0,1]）
        '''
        u = []
        d = []
        for i in label:
            if i:
                u += self._saw_tooth().tolist()
            else:
                u += self._sinusoidal().tolist()
            d += self._make_output(i).tolist()
        return np.array(u), np.array(d)

    def _sinusoidal(self):
        """ 正弦波 """
        t = np.linspace(0., self.period, self.period_length)
        x = np.sin(2. * np.pi * self.freq * t)
        return x

    def _saw_tooth(self):
        """ のこぎり波 """
        t = np.linspace(0., self.period, self.period_length)
        x = 2. * (t / self.period - np.floor(t / self.period + 0.5))
        return x

    def _make_output(self, label):
        y = np.zeros((self.period_length, 2))
        y[:, label] = 1
        return y
