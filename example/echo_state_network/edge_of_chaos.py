import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from argparse import ArgumentParser

from dh_snnkit.echo_state_network.network_maker import uniform


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument('--sim_time', type=float, default=10.)
    parser.add_argument('--trans_time', type=float, default=1.)
    parser.add_argument('--freq', type=float, default=1.)
    parser.add_argument('--max_rho', type=float, default=2.)
    parser.add_argument('--num_rho', type=int, default=20)
    parser.add_argument('--num_plot', type=int, default=3)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    rnd.seed(args.seed)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # 時系列データ生成
    time = np.arange(0., args.sim_time, args.dt)
    u_list = np.sin(2. * np.pi * args.freq * time)
    sim_steps = time.shape[0]
    trans_steps = int(args.trans_time / args.dt)
    period_steps = int(1. / (args.freq * args.dt))

    # スペクトル半径rhoの値を変えながらループ
    p_all = []
    rho_list = np.linspace(0., args.max_rho, args.num_rho)
    for rho in tqdm(rho_list):

        # 入力層とリザバーを生成
        config['layer']['rho'] = rho
        input, _, _, reservoir = uniform(**config['layer'])

        # リザバー状態の時間発展
        x_all = []
        for t, u in zip(time, u_list):
            x_in = input(np.array([u]))
            x = reservoir.step(x_in, args.dt)
            x_all.append(x.copy())
        x_all = np.array(x_all)

        # 1周期おきの状態
        period_x = x_all[trans_steps:sim_steps:period_steps, 0:args.num_plot]
        p = np.hstack((
            np.full((period_x.shape[0], 1), rho),
            period_x,
        ))
        p_all += p.tolist()
    p_all = np.array(p_all)

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 7))
    plt.subplots_adjust(hspace=0.3)

    for i in range(0, args.num_plot):
        ax = fig.add_subplot(3, 1, i + 1)
        plt.scatter(p_all[:, 0], p_all[:, i + 1], color='k', marker='o', s=5)
        plt.ylabel(f'p_{i + 1}')
        plt.grid(True)
        if i == args.num_plot - 1:
            plt.xlabel(r'$\rho$')

    plt.show()
