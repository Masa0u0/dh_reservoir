# 様々な周波数で構成された多次元波の分類タスク

import argparse
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
from tqdm import tqdm
from time import time
from rich import print

from dh_function.metrics import cross_entropy_error
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
    parser.add_argument("--sim_time", type=float, default=5.)
    parser.add_argument("--plot_time", type=float, default=1.)
    parser.add_argument("--start_time", type=float, default=0.5)
    parser.add_argument("--freq_range", type=float, nargs=2, default=(1., 10.))
    parser.add_argument("--delay_range", type=float, nargs=2, default=(0., 3.))
    parser.add_argument("--num_class", type=int, default=100)
    parser.add_argument("--max_plot_class", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=str)
    parser.add_argument("--encoder_param_path", type=str)
    parser.add_argument("--lsm_param_path", type=str)
    args = parser.parse_args()

    assert args.max_plot_class <= args.num_class
    assert args.plot_time <= args.sim_time
    assert args.config or (args.encoder_param_path and args.lsm_param_path)
    np.random.seed(args.seed)

    sim_steps = int(args.sim_time / args.dt)
    plot_steps = int(args.plot_time / args.dt)
    start_steps = int(args.start_time / args.dt)

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

    obs_dim = encoder.obs_dim

    # データセットの作成
    X = []
    y = []
    num_class_plot = min(args.max_plot_class, args.num_class)
    obs_buf = np.empty((num_class_plot, obs_dim, plot_steps))
    start_time = time()

    for cls in tqdm(range(args.num_class)):
        # 入力信号生成用パラメータ
        freq = np.random.uniform(args.freq_range[0], args.freq_range[1], (obs_dim,))
        delay = np.random.uniform(args.delay_range[0], args.delay_range[1], (obs_dim,))
        target = np.zeros((args.num_class,))
        target[cls] = 1.

        lsm.reset()
        for step in range(sim_steps):
            t = step * args.dt + delay
            obs = np.sin(2. * np.pi * freq * t)
            if cls < num_class_plot and step < plot_steps:
                obs_buf[cls, :, step] = obs
            input_spike = encoder.encode(obs, args.dt)
            rsrvr_trace = lsm.step(input_spike, args.dt)

            if step > start_steps:
                X.append(rsrvr_trace)
                y.append(target)

    end_time = time()
    elapsed_time = end_time - start_time
    time_for_1ms_sim = elapsed_time / (args.num_class * args.sim_time)

    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y))
    # print(np.argmax(y_train[:20, :], axis=1))

    # 学習
    model = MLPClassifier(hidden_layer_sizes=())
    model.fit(X_train, y_train)

    # 評価
    y_pred = model.predict_proba(X_test)
    acc = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
    cer = cross_entropy_error(y_test, y_pred)

    # 表示
    print(
        f'accuray: {acc:.3f}, '
        f'cross_entropy_error: {cer:.3f} '
        f'time_for_1ms_sim: {time_for_1ms_sim:.3f}'
    )

    # 入力信号の描画
    fig = plt.figure()
    t_array = np.arange(0, plot_steps) * args.dt
    for i in range(num_class_plot):
        ax = fig.add_subplot(num_class_plot, 1, i + 1)
        if i == num_class_plot - 1:
            ax.set_xlabel("Time [s]")
        for j in range(0, obs_dim):
            ax.plot(t_array, obs_buf[i, j, :])
        ax.set_ylim(-1., 1.)
    plt.show()
