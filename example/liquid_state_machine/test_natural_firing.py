# 自然発火のシミュレーション

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from rich import print

from dh_reservoir.liquid_state_machine import LiquidStateMachine
from dh_reservoir.liquid_state_machine.network_maker import ponghiran2019, klampfl2013


network_maker_dict = {
    "ponghiran2019": ponghiran2019,
    "klampfl2013": klampfl2013,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--sim_time", type=float, default=10.)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=str)
    parser.add_argument("--lsm_param_path", type=str)
    args = parser.parse_args()

    assert args.config or (args.encoder_param_path and args.lsm_param_path)
    np.random.seed(args.seed)

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

    if args.lsm_param_path is None:
        lsm_param = network_maker_dict[config["network_maker"]](**config["LSM"])
        lsm = LiquidStateMachine(param=lsm_param, seed=args.seed)
    else:
        lsm = LiquidStateMachine(load_path=args.lsm_param_path, seed=args.seed)

    print(f'入力層の結合率    : {lsm.input_connectivity * 100.:.3f}%')
    print(f'リザバー層の結合率: {lsm.rsrvr_connectivity * 100.:.3f}%')

    sim_steps = int(args.sim_time / args.dt)

    # シミュレーション
    lsm.start_log()
    start_time = time()
    for step in tqdm(range(sim_steps)):
        input_spike = np.zeros((lsm.num_input,))
        lsm.step(input_spike, args.dt)
    end_time = time()
    print(f'1msのシミュレーションにかかった時間: {round((end_time - start_time) / args.sim_time, 2)}ms')

    lsm.plot_eig()
    lsm.plot_spike_log(args.dt)
    plt.show()
