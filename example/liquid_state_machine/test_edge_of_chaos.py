# カオスの縁

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dh_snnkit.encoder import LIFEncoder
from dh_snnkit.liquid_state_machine import LiquidStateMachine
from dh_snnkit.liquid_state_machine.network_maker import ponghiran2019, klampfl2013


network_maker_dict = {
    "ponghiran2019": ponghiran2019,
    "klampfl2013": klampfl2013,
}

tau_syn_mean_dict_base = {"ee": 0.006, "ei": 0.006, "ie": 0.012, "ii": 0.012}
tau_syn_sd_dict_base = {"ee": 0.003, "ei": 0.003, "ie": 0.006, "ii": 0.006}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sim_time", type=float, default=10.)
    parser.add_argument("--start_time", type=float, default=1.)
    parser.add_argument("--freq", type=float, default=10.)
    parser.add_argument("--tau_amp_range", type=float, nargs=2, default=(1., 5.))
    parser.add_argument("--num_tau_amp", type=int, default=30)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

    sim_steps = int(args.sim_time / args.dt)
    start_steps = int(args.start_time / args.dt)
    sampling_interval = int(1. / (args.freq * args.dt))
    encoder = LIFEncoder(**config["Encoder"])

    delta = np.random.uniform(0., 100., (config["Encoder"]["obs_dim"],))
    trace_std_buf = []
    tau_amp_list = np.linspace(args.tau_amp_range[0], args.tau_amp_range[1], args.num_tau_amp)

    for tau_amp in tqdm(tau_amp_list):
        tau_syn_mean_dict = {key: val * tau_amp for key, val in tau_syn_mean_dict_base.items()}
        tau_syn_sd_dict = {key: val * tau_amp for key, val in tau_syn_sd_dict_base.items()}
        lsm_param = network_maker_dict[config["network_maker"]](
            tau_syn_mean_dict=tau_syn_mean_dict,
            tau_syn_sd_dict=tau_syn_sd_dict,
            **config["LSM"],
        )
        lsm = LiquidStateMachine(param=lsm_param)
        encoder.reset()
        lsm.reset()
        trace_buf = []

        # シミュレーション
        for step in range(sim_steps):
            t = step * args.dt
            obs = np.sin(2. * np.pi * args.freq * (t + delta))
            input_spike = encoder.encode(obs, args.dt)
            lsm.step(input_spike, args.dt)

            if step > start_steps and step % sampling_interval == 0:
                trace_buf.append(lsm.rsrvr_trace)

        trace_std_buf.append(np.mean(np.std(trace_buf, axis=0)))

    plt.figure(figsize=(9., 9.))
    plt.xlabel("amplitude of synapse time constants")
    plt.ylabel("Trace STD")
    plt.title(f'{args.freq}Hz ✕ {encoder.obs_dim}')
    plt.scatter(tau_amp_list, trace_std_buf)
    plt.show()
