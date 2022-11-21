import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
from rich import print
from tqdm import tqdm
from time import time

from dh_snnkit.ext_lsm import LiquidStateMachine, NeuronParams, SynapseParams
from dh_snnkit.encoder import PoissonEncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--max_input_fire_rate", type=float, default=30.)
    parser.add_argument("--max_wta_fire_rate", type=float, default=100.)
    parser.add_argument("--connectivity", type=float, default=0.5)
    parser.add_argument("--lambda_input", type=float, default=10.)
    parser.add_argument("--lambda_rsrvr", type=float, default=2.)
    parser.add_argument("--wta_neurons_range", type=tuple, default=(10, 10))
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--train_time", type=float, default=10.)
    parser.add_argument("--test_time", type=float, default=1.)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--num_input_neurons", type=int, default=10)
    parser.add_argument("--num_wta", type=int, default=50)
    parser.add_argument("--frame_skip", type=int, default=30)
    parser.add_argument("--pred_frames", type=int, default=30)
    args = parser.parse_args()

    if args.seed >= 0:
        np.random.seed(args.seed)

    radius_in = 5.
    radius_re = 4.

    n_in = args.num_input_neurons

    # ポアソンエンコーダを定義
    encoder = PoissonEncoder(obs_dim=n_in, max_freq=args.max_input_fire_rate)

    # 入力ニューロンの位置
    theta_in = np.random.uniform(low=0., high=2 * np.pi, size=(n_in,))
    phi_in = np.random.uniform(low=0., high=2 * np.pi, size=(n_in,))
    pos_in = radius_in * np.c_[
        np.sin(theta_in) * np.cos(phi_in),
        np.sin(theta_in) * np.sin(phi_in),
        np.cos(theta_in),
    ]

    # WTA回路の位置
    pos_wta = np.empty((args.num_wta, 3))
    for i in range(0, args.num_wta):
        while True:
            cnd = np.random.uniform(low=-radius_re, high=radius_re, size=(3,))
            if np.sum(cnd**2) < radius_re**2:
                pos_wta[i, :] = cnd
                break

    # リキッドステートマシンを作成
    lsm = LiquidStateMachine(
        pos_input=pos_in,
        pos_wta=pos_wta,
        neuron_params=NeuronParams(),
        synapse_params=SynapseParams(),
        wta_neurons_range=args.wta_neurons_range,
        connectivity=args.connectivity,
        lambda_input=args.lambda_input,
        lambda_rsrvr=args.lambda_rsrvr,
        stochastic=args.stochastic,
        max_fire_rate=args.max_wta_fire_rate,
        seed=args.seed,
    )
    print(f'入力層のニューロン数: {lsm.num_input_neurons}')
    print(f'リザバー層のニューロン数: {lsm.num_rsrvr_neurons}')
    print(f'入力層の結合率    : {lsm.input_connectivity * 100}%')
    print(f'リザバー層の結合率: {lsm.rsrvr_connectivity * 100}%')

    # リードアウト(隠れ層をなしにしたら単純パーセプトロンになる？)
    readout = MLPRegressor(hidden_layer_sizes=())

    # 学習
    spike_log = np.zeros((n_in, args.pred_frames))   # 直近pred_framesフレームのスパイクを記録する配列
    t_train = []   # 誤差の横軸用
    error_log = []   # 誤差の記録
    start_time = time()
    for step in tqdm(range(int(args.train_time / args.dt))):
        x = np.random.rand(n_in)
        s_in = encoder.encode(x, args.dt)
        rsrvr_traces = np.array(lsm.step(s_in, args.dt)).reshape(1, -1)
        # print(rsrvr_traces)
        spike_log = np.c_[spike_log[:, 1:], s_in]
        if step % args.frame_skip == args.frame_skip - 1:
            input_layer_spike_num = np.sum(spike_log)   # 入力層のニューロンにおける、pred_range内の総スパイク数
            readout.partial_fit(rsrvr_traces, np.array([input_layer_spike_num]))   # オンライン学習
            t_train.append(step * args.dt)
            error_log.append(abs(input_layer_spike_num - readout.predict(rsrvr_traces)[0]))
    end_time = time()
    print(f'1msのテストにかかった時間: {round((end_time - start_time) / args.train_time, 2)}ms')

    # テスト
    lsm.reset()
    # lsm.start_log()
    t_test = []
    true_list = []
    pred_list = []
    spike_log *= 0
    for step in tqdm(range(int(args.test_time / args.dt))):
        x = np.random.rand(n_in)
        s_in = encoder.encode(x, args.dt)
        rsrvr_traces = np.array(lsm.step(s_in, args.dt)).reshape(1, -1)
        spike_log = np.c_[spike_log[:, 1:], s_in]
        if step % args.frame_skip == args.frame_skip - 1:
            input_layer_spike_num = np.sum(spike_log)   # 入力層のニューロンにおける、pred_range内の総スパイク数
            pred = readout.predict(rsrvr_traces)[0]
            t_test.append(step * args.dt)
            true_list.append(input_layer_spike_num)
            pred_list.append(pred)

    # 描画
    # lsm.show_spike_log()

    plt.plot(t_train, error_log)
    plt.title("Error")
    plt.show()

    plt.plot(t_test, true_list, label="True")
    plt.plot(t_test, pred_list, label="Pred")
    plt.legend()
    plt.title("Comparison")
    plt.show()
