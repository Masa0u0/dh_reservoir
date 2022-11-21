import numpy as np
import pickle
from tqdm import tqdm

from .layer import Input, Output, Feedback, Reservoir


class EchoStateNetwork:

    def __init__(
        self,
        input: Input = None,
        output: Output = None,
        feedback: Feedback = None,
        reservoir: Reservoir = None,
        noise_level: float = 0.,
        load_path: str = None,
    ):
        if load_path is None:
            assert input is not None
            assert output is not None
            assert reservoir is not None

            self.Input = input
            self.Output = output
            self.Feedback = feedback
            self.Reservoir = reservoir
            self.noise_level = noise_level
        else:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)

            self.Input = data['Input']
            self.Output = data['Output']
            self.Feedback = data['Feedback']
            self.Reservoir = Reservoir(**data['rsrvr_attributes'])
            self.noise_level = data['noise_level']

        self.N_u = self.Input.N_u
        self.N_y = self.Output.N_y
        self.N_x = self.Reservoir.N_x
        self.y_prev = np.zeros((self.N_y,))

    @property
    def rsrvr_state(self):
        return self.Reservoir.get_state()

    def step(self, u, dt):
        x_in = self.Input(u)

        # フィードバック結合
        if self.Feedback is not None:
            x_back = self.Feedback(self.y_prev)
            x_in += x_back

        # ノイズ
        if self.noise_level > 0.:
            x_in += np.random.uniform(-self.noise_level, self.noise_level, (self.N_x,))

        # リザバー状態ベクトル
        return self.Reservoir.step(x_in, dt)

    def reset(self):
        self.Reservoir.reset()
        self.y_prev *= 0.

    def save(self, path: str):
        assert path.endswith('.pkl')

        data = {
            'Input': self.Input,
            'Output': self.Output,
            'Feedback': self.Feedback,
            'rsrvr_attributes': self.Reservoir.get_attributes(),
            'noise_level': self.noise_level,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    # バッチ学習
    def train(self, U, D, optimizer, dt, trans_len=None):
        '''
        U: 教師データの入力, データ長×N_u
        D: 教師データの出力, データ長×N_y
        optimizer: 学習器
        trans_len: 過渡期の長さ
        return: 学習前のモデル出力, データ長×N_y
        '''
        assert U.ndim == D.ndim == 2
        assert U.shape[0] == D.shape[0]
        assert U.shape[1] == self.N_u and D.shape[1] == self.N_y

        train_len = len(U)
        if trans_len is None:
            trans_len = 0
        Y = []

        # 時間発展
        for n in tqdm(range(train_len)):
            x = self.step(U[n], dt)

            # 目標値
            d = D[n]

            # 学習器
            if n > trans_len:  # 過渡期を過ぎたら
                optimizer(d, x)

            # 学習前のモデル出力
            y = self.Output(x)
            Y.append(y)
            self.y_prev = d

        # 学習済みの出力結合重み行列を設定
        self.Output.set_weight(optimizer.get_Wout_opt())

        # モデル出力（学習前）
        return np.array(Y)

    # バッチ学習後の予測
    def predict(self, U, dt):
        test_len = len(U)
        Y_pred = []

        # 時間発展
        for n in tqdm(range(test_len)):
            x_in = self.Input(U[n])

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            # リザバー状態ベクトル
            x = self.Reservoir.step(x_in, dt)

            # 学習後のモデル出力
            y_pred = self.Output(x)
            Y_pred.append(y_pred)
            self.y_prev = y_pred

        # モデル出力（学習後）
        return np.array(Y_pred)

    # バッチ学習後の予測（自律系のフリーラン）
    def run(self, U, dt):
        test_len = len(U)
        Y_pred = []
        y = U[0]

        # 時間発展
        for _ in tqdm(range(test_len)):
            x_in = self.Input(y)

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            # リザバー状態ベクトル
            x = self.Reservoir.step(x_in, dt)

            # 学習後のモデル出力
            y_pred = self.Output(x)
            Y_pred.append(y_pred)
            y = y_pred
            self.y_prev = y

        return np.array(Y_pred)

    # オンライン学習と予測
    def adapt(self, U, D, optimizer, dt):
        Y_pred = []
        Wout_abs_mean = []

        # 出力結合重み更新
        for n in tqdm(range(len(U))):
            x_in = self.Input(U[n])
            x = self.Reservoir.step(x_in, dt)
            d = D[n]

            # 学習
            Wout = optimizer(d, x)

            # モデル出力
            y = np.dot(Wout, x)
            Y_pred.append(y)
            Wout_abs_mean.append(np.mean(np.abs(Wout)))

        return np.array(Y_pred), np.array(Wout_abs_mean)
