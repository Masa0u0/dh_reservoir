import numpy as np
import numpy.linalg as LA
from numpy.typing import NDArray


# Moore-Penrose擬似逆行列
class Pseudoinv:

    def __init__(self, N_x, N_y) -> None:
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        '''
        self.X = np.empty((N_x, 0))
        self.D = np.empty((N_y, 0))

    # 状態集積行列および教師集積行列の更新
    def __call__(self, d, x) -> None:
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X = np.hstack((self.X, x))
        self.D = np.hstack((self.D, d))

    # Woutの最適解（近似解）の導出
    def get_Wout_opt(self) -> NDArray:
        Wout_opt = np.dot(self.D, LA.pinv(self.X))
        return Wout_opt


# リッジ回帰（beta=0のときは線形回帰）
class Tikhonov:

    def __init__(self, N_x, N_y, beta) -> None:
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param beta: 正則化パラメータ
        '''
        self.beta = beta
        self.X_XT = np.zeros((N_x, N_x))
        self.D_XT = np.zeros((N_y, N_x))
        self.N_x = N_x

    # 学習用の行列の更新
    def __call__(self, d, x) -> None:
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X_XT += np.dot(x, x.T)
        self.D_XT += np.dot(d, x.T)

    # Woutの最適解（近似解）の導出
    def get_Wout_opt(self) -> NDArray:
        X_pseudo_inv = LA.inv(self.X_XT + self.beta * np.identity(self.N_x))
        Wout_opt = np.dot(self.D_XT, X_pseudo_inv)
        return Wout_opt


# 逐次最小二乗(RLS)法
class RLS:

    def __init__(self, N_x, N_y, delta, lam, update) -> None:
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param delta: 行列Pの初期条件の係数（P=delta*I, 0<delta<<1）
        param lam: 忘却係数 (0<lam<1, 1に近い値)
        param update: 各時刻での更新繰り返し回数
        '''
        self.delta = delta
        self.lam = lam
        self.update = update
        self.P = (1. / self.delta) * np.eye(N_x, N_x)
        self.Wout = np.zeros([N_y, N_x])

    # Woutの更新
    def __call__(self, d, x) -> NDArray:
        x = np.reshape(x, (-1, 1))
        for _ in np.arange(self.update):
            v = d - np.dot(self.Wout, x)
            gain = (1 / self.lam * np.dot(self.P, x))
            gain = gain / (1 + 1 / self.lam * np.dot(np.dot(x.T, self.P), x))
            self.P = 1 / self.lam * (self.P - np.dot(np.dot(gain, x.T), self.P))
            self.Wout += np.dot(v, gain.T)

        return self.Wout
