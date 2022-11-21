import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

from dh_reservoir.echo_state_network.network_maker import uniform


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dt', type=float, default=1e-3)
    parser.add_argument('--sim_time', type=float, default=10.)
    parser.add_argument('--freq', type=float, default=1.)
    parser.add_argument('--num_plot', type=int, default=3)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # 時系列データ生成
    time = np.arange(0., args.sim_time, args.dt)
    u_list = np.sin(2. * np.pi * args.freq * time)

    # 入力層とリザバーの生成
    input, _, _, reservoir = uniform(**config['layer'])
    assert input.N_u == 1

    # リザバー状態の時間発展
    x_all = []
    for t, u in tqdm(zip(time, u_list)):
        x_in = input(np.array([u]))
        x = reservoir.step(x_in, args.dt)
        x_all.append(x.copy())
    x_all = np.array(x_all)

    # グラフ表示
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, args.num_plot, wspace=0.4, hspace=0.4)

    for i in range(0, args.num_plot):
        ax1 = plt.subplot(gs[0, i])
        ax1.plot(time, x_all[:, i], color='k', linewidth=2)
        ax1.set_xlabel('time[s]')
        ax1.set_ylabel(f'x_{i + 1}')
        ax1.set_ylim(-1, 1)
        ax1.grid(True)

        ax2 = plt.subplot(gs[1, i])
        ax2.plot(u_list, x_all[:, i], color='k', linewidth=2)
        ax2.set_xlabel('u')
        ax2.set_ylabel(f'x_{i + 1}')
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.grid(True)

    plt.show()
