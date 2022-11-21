import argparse
import pickle
import yaml
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from rich import print
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from dh_function.metrics import root_mean_squared_error
from dh_reservoir.echo_state_network import EchoStateNetwork


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sim_time", type=float, default=10.)
    parser.add_argument("--plot_time", type=float)
    parser.add_argument("--start_time", type=float, default=0.5)
    parser.add_argument("--freq_range", type=float, nargs=2, default=(1., 10.))
    parser.add_argument("--delay_range", type=float, nargs=2, default=(0., 3.))
    parser.add_argument("--noise_scale", type=float, default=0.)
    parser.add_argument("--num_plot", type=int, default=10**10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--config", type=str)
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    assert (args.config is None) ^ (args.load_path is None)
    if args.plot_time:
        assert args.plot_time <= args.sim_time

    np.random.seed(args.seed)

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

    if args.load_path:
        with open(args.load_path, 'rb') as f:
            esn = pickle.load(f)
    else:
        esn = EchoStateNetwork(**config)

    sim_steps = int(args.sim_time / args.dt)
    plot_time = args.plot_time if args.plot_time else args.sim_time
    plot_steps = int(plot_time / args.dt)
    start_steps = int(args.start_time / args.dt)
    obs_dim = esn.N_u

    # データセットの作成
    X = []
    y = []
    num_plot = min(args.num_plot, obs_dim)
    obs_buf = np.empty((num_plot, plot_steps))
    target_buf = np.empty((num_plot, plot_steps))
    freq = np.random.uniform(args.freq_range[0], args.freq_range[1], (obs_dim,))
    delay = np.random.uniform(args.delay_range[0], args.delay_range[1], (obs_dim,))
    esn.reset()

    start_time = time()

    for step in tqdm(range(sim_steps)):
        t = step * args.dt + delay
        c = 2. * np.pi * freq
        x = c * t
        sx = np.sqrt(x)
        sin = np.sin(x)
        cos = np.cos(x)
        ssin = np.sin(sx)
        scos = np.cos(sx)

        obs = sin * ssin + np.random.uniform(-args.noise_scale, args.noise_scale, (obs_dim,))
        target = c * (ssin * cos + sin * scos / (2. * sx))

        rsrvr_state = esn.step(obs)

        if step < plot_steps:
            obs_buf[:, step] = obs[:num_plot]
            target_buf[:, step] = target[:num_plot]

        if step > start_steps:
            X.append(rsrvr_state)
            y.append(target)

    end_time = time()
    print(f'1msのシミュレーションにかかった時間: {round((end_time - start_time) / args.sim_time, 2)}ms')

    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y))
    print(f'the number of train data: {X_train.shape[0]}')
    print(f'the number of test data : {X_test.shape[0]}')

    # 線型回帰モデルの定義
    model = LinearRegression()

    # 学習
    model.fit(X_train, y_train)

    # 予測
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # 平方平均二乗誤差
    print(f'Train RMSE: {root_mean_squared_error(y_train, pred_train)}')
    print(f'Test RMSE: {root_mean_squared_error(y_test, pred_test)}')

    # 決定係数
    print(f'Train R2: {r2_score(y_train, pred_train)}')
    print(f'Test R2: {r2_score(y_test, pred_test)}')

    # モデル保存
    if args.save_path:
        model.save(args.save_path)

    # 描画
    if args.plot:
        t_array = np.arange(0, plot_steps) * args.dt

        # 入力信号
        fig = plt.figure()
        plt.title("Input")
        for i in range(num_plot):
            ax = fig.add_subplot(num_plot, 1, i + 1)
            ax.plot(t_array, obs_buf[i, :])
            lim = max(1., args.noise_scale)
            ax.set_ylim(-lim, lim)
            ax.text(-0.15, 0.5, f'{freq[i]:.1f}Hz', transform=ax.transAxes)
            if i == num_plot - 1:
                ax.set_xlabel("Time [s]")

        # 出力信号
        fig = plt.figure()
        plt.title("Target")
        for i in range(num_plot):
            ax = fig.add_subplot(num_plot, 1, i + 1)
            ax.plot(t_array, target_buf[i, :])
            ax.text(-0.15, 0.5, f'{freq[i]:.1f}Hz', transform=ax.transAxes)
            if i == num_plot - 1:
                ax.set_xlabel("Time [s]")

        plt.show()
