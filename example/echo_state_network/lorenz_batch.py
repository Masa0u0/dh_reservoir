import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from dh_reservoir.echo_state_network import EchoStateNetwork
from dh_reservoir.echo_state_network.optimizer import Tikhonov
from dh_reservoir.echo_state_network.network_maker import uniform
from dh_reservoir.echo_state_network.data_generator import LorenzAttractor


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--a', type=float, default=10.)
    parser.add_argument('--b', type=float, default=28.)
    parser.add_argument('--c', type=float, default=8./3.)
    parser.add_argument('--T', type=float, default=100.)
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--N_x', type=int, default=50)
    parser.add_argument('--input_scale', type=float, default=1.)
    parser.add_argument('--fb_scale', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=1.)
    parser.add_argument('--indegree', type=float, default=7.5)
    parser.add_argument('--rho', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=1e-4)
    args = parser.parse_args()

    # データ生成
    dynamics = LorenzAttractor(args.a, args.b, args.c)
    u, d = dynamics.generate_data(label=1., T=args.T, dt=args.dt)

    # 学習・テスト用情報
    data_length = u.shape[0]
    train_length = int(data_length * args.train_ratio)
    train_U = u[:train_length, np.newaxis]
    train_D = d[:train_length, :]
    test_U = u[train_length:, np.newaxis]
    test_D = d[train_length:, :]

    # ESNモデル
    input_, output, feedback, reservoir = uniform(
        N_u=1,
        N_y=3,
        N_x=args.N_x,
        input_scale=args.input_scale,
        fb_scale=args.fb_scale,
        tau=args.tau,
        indegree=args.indegree,
        rho=args.rho,
    )
    esn = EchoStateNetwork(
        input=input_,
        output=output,
        feedback=feedback,
        reservoir=reservoir,
    )

    # 学習（リッジ回帰）
    train_Y = esn.train(
        U=train_U,
        D=train_D,
        optimizer=Tikhonov(args.N_x, train_D.shape[1], args.beta),
    )

    # モデル出力
    test_Y = esn.predict(test_U)

    # 評価（テスト誤差RMSE, NRMSE）
    rmse = np.sqrt(((test_D - test_Y) ** 2).mean())
    nrmse = rmse / np.sqrt(np.var(test_D))
    print('RMSE =', rmse)
    print('NRMSE =', nrmse)

    # グラフ表示
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Lorenz')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    x_target = test_D[:, 0]
    y_target = test_D[:, 1]
    z_target = test_D[:, 2]
    x_pred = test_Y[:, 0]
    y_pred = test_Y[:, 1]
    z_pred = test_Y[:, 2]
    ax.plot(x_target, y_target, z_target, color='red', label='target')
    ax.plot(x_pred, y_pred, z_pred, color='blue', label='predicted')
    ax.legend()
    plt.show()
