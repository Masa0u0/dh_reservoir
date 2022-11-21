# 周波数応答

import os
import os.path as osp
import argparse
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from rich import print

from dh_snnkit.encoder import LIFEncoder
from dh_snnkit.liquid_state_machine import LiquidStateMachine
from dh_snnkit.liquid_state_machine.network_maker import ponghiran2019, klampfl2013


network_maker_dict = {
    "ponghiran2019": ponghiran2019,
    "klampfl2013": klampfl2013,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--sim_time", type=float, default=3.)
    parser.add_argument("--freq", type=float, default=10.)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--show_input_waves", action="store_true")
    parser.add_argument("--config", type=str)
    parser.add_argument("--encoder_param_path", type=str)
    parser.add_argument("--lsm_param_path", type=str)
    parser.add_argument("--save_dir", type=str)
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

    print(f'リザバー層のニューロン数: {lsm.num_rsrvr}')
    print(f'入力層の結合率          : {lsm.input_connectivity * 100.:.3f}%')
    print(f'リザバー層の結合率      : {lsm.rsrvr_connectivity * 100.:.3f}%')

    obs_dim = encoder.obs_dim
    obs_buf = []
    sim_steps = int(args.sim_time / args.dt)
    delta = np.random.uniform(0., 100., (obs_dim,))

    # シミュレーション
    lsm.start_log()
    start_time = time()
    for step in tqdm(range(sim_steps)):
        t = step * args.dt
        obs = np.sin(2. * np.pi * args.freq * t + delta)
        obs_buf.append(obs)
        input_spike = encoder.encode(obs, args.dt)
        lsm.step(input_spike, args.dt)
    end_time = time()
    print(f'1msのシミュレーションにかかった時間: {round((end_time - start_time) / args.sim_time, 2)}ms')

    # モデルの保存
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(osp.join(args.save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        encoder.save(osp.join(args.save_dir, "encoder_params.json"))
        lsm.save(osp.join(args.save_dir, "lsm_params.pkl"))

    # 入力信号の描画
    if args.show_input_waves:
        plt.figure(figsize=(12, 9))
        plt.xlabel("Time [s]")
        plt.title("Input Sin Waves")
        plt.ylim(-1., 1.)
        t_array = np.arange(0, sim_steps) * args.dt
        for obs in obs_buf:
            plt.plot(t_array, obs, linewidth=0.7)

    lsm.plot_eig()
    lsm.plot_spike_log(args.dt)
    plt.show()
