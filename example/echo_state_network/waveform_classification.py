import os
import os.path as osp
import yaml
import json
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from dh_snnkit.echo_state_network import EchoStateNetwork
from dh_snnkit.echo_state_network.util import Tikhonov
from dh_snnkit.echo_state_network.network_maker import uniform


class SinSaw:
    """ 正弦波とのこぎり波の混合波形生成 """

    def __init__(self, freq: float, dt: float):
        self.freq = freq
        self.period = 1. / freq
        self.dt = dt
        self.period_length = int(self.period / self.dt)
        assert self.period_length > 0

    def sinusoidal(self):
        """ 正弦波 """
        t = np.linspace(0., self.period, self.period_length)
        x = np.sin(2. * np.pi * self.freq * t)
        return x

    def saw_tooth(self):
        """ のこぎり波 """
        t = np.linspace(0., self.period, self.period_length)
        x = 2. * (t / self.period - np.floor(t / self.period + 0.5))
        return x

    def make_output(self, label):
        y = np.zeros((self.period_length, 2))
        y[:, label] = 1
        return y

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
                u += self.saw_tooth().tolist()
            else:
                u += self.sinusoidal().tolist()
            d += self.make_output(i).tolist()
        return np.array(u), np.array(d)


class ScalingShift:
    """ 出力のスケーリング """

    def __init__(self, scale, shift):
        '''
        :param scale: 出力層のスケーリング（scale[n]が第n成分のスケーリング）
        :param shift: 出力層のシフト（shift[n]が第n成分のシフト）
        '''
        self.scale = np.diag(scale)
        self.shift = np.array(shift)
        self.inv_scale = LA.inv(self.scale)
        self.inv_shift = -np.dot(self.inv_scale, self.shift)

    def __call__(self, x):
        return np.dot(self.scale, x) + self.shift

    def inverse(self, x):
        return np.dot(self.inv_scale, x) + self.inv_shift


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument('--n_wave_train', type=int, default=10)
    parser.add_argument('--n_wave_test', type=int, default=5)
    parser.add_argument('--n_wave_plot', type=int, default=5)
    parser.add_argument("--freq", type=float, default=1.)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--config', type=str)
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    assert (args.config is None) ^ (args.load_path is None)
    assert 0 < args.n_wave_plot <= args.n_wave_test <= args.n_wave_train

    np.random.seed(args.seed)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # 時系列入力データ生成
    dynamics = SinSaw(args.freq, args.dt)
    label = np.random.choice(2, args.n_wave_train + args.n_wave_test)
    u, d = dynamics.generate_data(label)
    T = dynamics.period_length * args.n_wave_train

    # 訓練・検証用情報
    train_U = u[:T].reshape(-1, 1)
    train_D = d[:T]

    test_U = u[T:].reshape(-1, 1)
    test_D = d[T:]

    # 出力のスケーリング関数
    output_func = ScalingShift([0.5, 0.5], [0.5, 0.5])

    # ESNモデル
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        input_, output, feedback, reservoir = uniform(**config['layer'])
        esn = EchoStateNetwork(
            input=input_,
            output=output,
            feedback=feedback,
            reservoir=reservoir,
            **config['other'],
        )
    else:
        esn = EchoStateNetwork(load_path=args.load_path)

    assert esn.N_u == 1 and esn.N_y == 2

    # 学習（リッジ回帰）
    train_Y = esn.train(train_U, train_D, Tikhonov(esn.N_x, train_D.shape[1], 0.1), args.dt)

    # 訓練データに対するモデル出力
    test_Y = esn.predict(test_U, args.dt)

    # 評価（正解率, accracy）
    mode = np.empty(0, np.int)
    for i in range(args.n_wave_test):
        tmp = test_Y[dynamics.period_length * i:dynamics.period_length * (i + 1), :]   # 各ブロックの出力
        max_index = np.argmax(tmp, axis=1)   # 最大値をとるインデックス
        histogram = np.bincount(max_index)   # そのインデックスのヒストグラム
        mode = np.hstack((mode, np.argmax(histogram)))   # 最頻値

    target = test_D[0:dynamics.period_length * args.n_wave_test:dynamics.period_length, 1]
    accuracy = 1 - LA.norm(mode.astype(float) - target, 1) / args.n_wave_test
    print('accuracy = ', accuracy)

    # モデル保存
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(osp.join(args.save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        esn.save(osp.join(args.save_dir, "esn.pkl"))

    # グラフ表示用データ
    plot_length = dynamics.period_length * args.n_wave_plot
    t_axis = np.arange(-plot_length, plot_length) * args.dt
    disp_U = np.concatenate((train_U[-plot_length:], test_U[:plot_length]))
    disp_D = np.concatenate((train_D[-plot_length:], test_D[:plot_length]))
    disp_Y = np.concatenate((train_Y[-plot_length:], test_Y[:plot_length]))

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 7))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.text(-0.1, 1, '(a)', transform=ax1.transAxes)
    ax1.text(0.2, 1.05, 'Training', transform=ax1.transAxes)
    ax1.text(0.7, 1.05, 'Testing', transform=ax1.transAxes)
    plt.plot(t_axis, disp_U[:, 0], color='k')
    plt.ylabel('Input')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.text(-0.1, 1, '(b)', transform=ax2.transAxes)
    plt.plot(t_axis, disp_D[:, 0], color='k', linestyle='-', label='Target')
    plt.plot(t_axis, disp_Y[:, 0], color='gray', linestyle='--', label='esn')
    plt.ylim([-0.3, 1.3])
    plt.ylabel('Output 1')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax3 = fig.add_subplot(3, 1, 3)
    plt.plot(t_axis, disp_D[:, 1], color='k', linestyle='-', label='Target')
    plt.plot(t_axis, disp_Y[:, 1], color='gray', linestyle='--', label='esn')
    plt.ylim([-0.3, 1.3])
    plt.xlabel('Time[s]')
    plt.ylabel('Output 2')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    plt.show()
