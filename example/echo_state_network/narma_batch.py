import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from dh_reservoir.echo_state_network import EchoStateNetwork
from dh_reservoir.echo_state_network.optimizer import Tikhonov
from dh_reservoir.echo_state_network.network_maker import uniform
from dh_reservoir.echo_state_network.data_generator import NARMA


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--N_x', type=int, default=50)
    parser.add_argument('--input_scale', type=float, default=0.1)
    parser.add_argument('--fb_scale', type=float, default=0.1)
    parser.add_argument('--indegree', type=float, default=7.5)  # N_x=50でdensity=0.15となるように設定
    parser.add_argument('--rho', type=float, default=0.9)
    parser.add_argument('--order', type=int, default=10, help='NARMAモデルの次数')
    parser.add_argument('--data_length_train', type=int, default=900, help='訓練用のデータ長')
    parser.add_argument('--data_length_test', type=int, default=900, help='評価用のデータ長')
    args = parser.parse_args()

    # データ生成
    dynamics = NARMA(args.order, a1=0.3, a2=0.05, a3=1.5, a4=0.1)
    data_length = args.data_length_train + args.data_length_test
    u, d = dynamics.generate_data(data_length)

    # 学習・テスト用情報
    train_U = u[:args.data_length_train].reshape(-1, 1)
    train_D = d[:args.data_length_train].reshape(-1, 1)
    test_U = u[args.data_length_train:].reshape(-1, 1)
    test_D = d[args.data_length_train:].reshape(-1, 1)

    # ESNモデル
    input_, output, feedback, reservoir = uniform(
        N_u=1,
        N_y=1,
        N_x=args.N_x,
        input_scale=args.input_scale,
        fb_scale=args.fb_scale,
        tau=0.,  # 即時アップデート
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
        optimizer=Tikhonov(args.N_x, train_D.shape[1], 1e-4),
    )

    # モデル出力
    test_Y = esn.predict(test_U)

    # 評価（テスト誤差RMSE, NRMSE）
    rmse = np.sqrt(((test_D - test_Y) ** 2).mean())
    nrmse = rmse/np.sqrt(np.var(test_D))
    print('RMSE =', rmse)
    print('NRMSE =', nrmse)

    # グラフ表示用データ
    T_disp = (-100, 100)
    t_axis = np.arange(T_disp[0], T_disp[1])
    disp_U = np.concatenate((train_U[T_disp[0]:], test_U[:T_disp[1]]))
    disp_D = np.concatenate((train_D[T_disp[0]:], test_D[:T_disp[1]]))
    disp_Y = np.concatenate((train_Y[T_disp[0]:], test_Y[:T_disp[1]]))

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 5))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    ax1.text(0.2, 1.05, 'Training', transform=ax1.transAxes)
    ax1.text(0.7, 1.05, 'Testing', transform=ax1.transAxes)
    plt.plot(t_axis, disp_U[:, 0], color='k')
    plt.ylabel('Input')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    plt.plot(t_axis, disp_D[:, 0], color='k', label='Target')
    plt.plot(t_axis, disp_Y[:, 0], color='gray', linestyle='--', label='Model')
    plt.xlabel('n')
    plt.ylabel('Output')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    plt.show()
