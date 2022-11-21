# ランダム発火の記憶・再現タスク(無理)

import argparse
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm

from dh_reservoir.encoder import LIFEncoder
from dh_reservoir.liquid_state_machine import LiquidStateMachine
from dh_reservoir.liquid_state_machine.network_maker import ponghiran2019, klampfl2013


network_maker_dict = {
    "ponghiran2019": ponghiran2019,
    "klampfl2013": klampfl2013,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sim_time", type=float, default=3.)
    parser.add_argument("--start_time", type=float, default=1.)
    parser.add_argument("--max_delay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str)
    parser.add_argument("--encoder_param_path", type=str)
    parser.add_argument("--lsm_param_path", type=str)
    args = parser.parse_args()

    assert args.config or (args.encoder_param_path and args.lsm_param_path)
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

    lsm.plot_eig()

    sim_steps = int(args.sim_time / args.dt)
    start_steps = int(args.start_time / args.dt)
    obs_buf = np.empty((encoder.obs_dim, 0))
    rsrvr_trace_buf = np.empty((lsm.num_rsrvr, 0))

    lsm.start_log()

    # シミュレーション
    for step in tqdm(range(sim_steps)):
        obs = np.random.uniform(-1., 1., (encoder.obs_dim,))
        input_spike = encoder.encode(obs, args.dt)
        rsrvr_trace = lsm.step(input_spike, args.dt)
        if step > start_steps:
            obs_buf = np.hstack((obs_buf, obs.reshape(-1, 1)))
            rsrvr_trace_buf = np.hstack((rsrvr_trace_buf, rsrvr_trace.reshape(-1, 1)))

    lsm.show_spike_log(args.dt)

    delay_time = np.arange(0., args.max_delay, 2e-3)
    delay_steps = (delay_time / args.dt).astype(int)
    max_delay_step = delay_steps[-1]
    X = rsrvr_trace_buf[:, max_delay_step:].T
    rmse_list = []

    # それぞれの遅延入力列に対して学習する
    for delay_step in tqdm(delay_steps):
        # データ作成
        y = np.roll(obs_buf, delay_step, axis=1)[:, max_delay_step:].T
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # 学習
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 評価
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_pred, y_test, squared=False)
        rmse_list.append(rmse)

    # 描画
    plt.figure()
    plt.title("Memory Capacity Test")
    plt.xlim(0., args.max_delay)
    # plt.ylim(0., 1.)
    plt.xlabel("delay time [s]")
    plt.ylabel("RMSE")
    plt.plot(delay_time, rmse_list)
    plt.show()
