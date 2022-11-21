# https://ieeexplore.ieee.org/document/1716628 を参考

import os
import os.path as osp
import yaml
import json
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from tqdm import tqdm
from rich import print

from dh_function.basic import calc_euclid_full_pairs
from dh_snnkit.encoder import LIFEncoder
from dh_snnkit.liquid_state_machine import LiquidStateMachine
from dh_snnkit.liquid_state_machine.network_maker import ponghiran2019, klampfl2013


network_maker_dict = {
    "ponghiran2019": ponghiran2019,
    "klampfl2013": klampfl2013,
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--sim_time", type=float, default=5.)
    parser.add_argument("--start_time", type=float, default=0.5)
    parser.add_argument("--freq_range", type=float, nargs=2, default=[0., 15.])
    parser.add_argument("--freq_step", type=float, default=1.)
    parser.add_argument("--noise_scale", type=float, default=0.)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--plot", action="store_true")
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

    print(f'入力層の結合率    : {lsm.input_connectivity * 100.:.3f}%')
    print(f'リザバー層の結合率: {lsm.rsrvr_connectivity * 100.:.3f}%')

    c_list = []
    t_list = np.arange(0., args.sim_time, args.dt)
    freq_list = np.arange(args.freq_range[0], args.freq_range[1], args.freq_step)
    for freq in tqdm(freq_list):
        rsrvr_trace_list = []
        x_list = np.sin(2. * np.pi * freq * t_list)
        lsm.reset()
        for t, x in zip(t_list, x_list):
            obs = x + np.random.uniform(-args.noise_scale, args.noise_scale, encoder.obs_dim)
            input_spike = encoder.encode(obs, args.dt)
            rsrvr_trace = lsm.step(input_spike, args.dt)
            if t > args.start_time:
                rsrvr_trace_list.append(rsrvr_trace.copy())
        c_list.append(np.mean(np.array(rsrvr_trace_list), axis=0))
    c_list = np.array(c_list)

    # Goodmanの分離度もどき
    sep = np.sqrt(
        np.sum(calc_euclid_full_pairs(c_list, c_list)**2) / (freq_list.size * lsm.num_rsrvr)**2
    )
    print(f'Goodman\'s score: {sep}')

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
        plt.show()
