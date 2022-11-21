# 非周期関数の微分を推測する

import os
import os.path as osp
import argparse
import yaml
import json
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from rich import print
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from dh_function.metrics import root_mean_squared_error
from dh_snnkit.encoder import LIFEncoder
from dh_snnkit.liquid_state_machine import LiquidStateMachine
from dh_snnkit.liquid_state_machine.network_maker import ponghiran2019, klampfl2013


network_maker_dict = {
    "ponghiran2019": ponghiran2019,
    "klampfl2013": klampfl2013,
}


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
    parser.add_argument("--encoder_param_path", type=str)
    parser.add_argument("--lsm_param_path", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    assert args.config or (args.encoder_param_path and args.lsm_param_path)
    if args.plot_time:
        assert args.plot_time <= args.sim_time

    np.random.seed(args.seed)

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

    if args.encoder_param_path is None:
        encoder = LIFEncoder(**config["Encoder"])
    else:
        encoder = LIFEncoder(load_path=args.encoder_param_path)

    if args.lsm_param_path is None:
        lsm_param = network_maker_dict[config["network_maker"]](**config["LSM"])
        lsm = LiquidStateMachine(param=lsm_param)
    else:
        lsm = LiquidStateMachine(load_path=args.lsm_param_path)

    print(f'入力層の結合率    : {lsm.input_connectivity * 100.:.3f}%')
    print(f'リザバー層の結合率: {lsm.rsrvr_connectivity * 100.:.3f}%')

    sim_steps = int(args.sim_time / args.dt)
    plot_time = args.plot_time if args.plot_time else args.sim_time
    plot_steps = int(plot_time / args.dt)
    start_steps = int(args.start_time / args.dt)
    obs_dim = encoder.obs_dim

    # データセットの作成
    X = []
    y = []
    num_plot = min(args.num_plot, obs_dim)
    obs_buf = np.empty((num_plot, plot_steps))
    target_buf = np.empty((num_plot, plot_steps))
    freq = np.random.uniform(args.freq_range[0], args.freq_range[1], (obs_dim,))
    delay = np.random.uniform(args.delay_range[0], args.delay_range[1], (obs_dim,))
    lsm.reset()
    lsm.start_log()

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

        input_spike = encoder.encode(obs, args.dt)
        input_spike *= 0.
        rsrvr_trace = lsm.step(input_spike, args.dt)

        if step < plot_steps:
            obs_buf[:, step] = obs[:num_plot]
            target_buf[:, step] = target[:num_plot]

        if step > start_steps:
            X.append(rsrvr_trace)
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

    # モデルの保存
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(osp.join(args.save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        encoder.save(osp.join(args.save_dir, "encoder_params.json"))
        lsm.save(osp.join(args.save_dir, "lsm_params.pkl"))

    # 描画
    if args.plot:
        lsm.plot_synapse_strength()
        lsm.plot_eig()
        lsm.plot_spike_log(dt=args.dt)

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
