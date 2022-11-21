# 入力ニューロンの時間窓内での総発火数を推測する

import argparse
import yaml
import numpy as np
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
from rich import print
from tqdm import tqdm
from time import time

from dh_reservoir.encoder import PoissonEncoder
from dh_reservoir.liquid_state_machine import LiquidStateMachine
from dh_reservoir.liquid_state_machine.network_maker import ponghiran2019, klampfl2013


network_maker_dict = {
    "ponghiran2019": ponghiran2019,
    "klampfl2013": klampfl2013,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--train_time", type=float, default=10.)
    parser.add_argument("--test_time", type=float, default=2.)
    parser.add_argument("--frame_skip", type=int, default=30)
    parser.add_argument("--pred_frames", type=int, default=30)
    parser.add_argument("--max_input_fire_rate", type=float, default=30.)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=str)
    parser.add_argument("--lsm_param_path", type=str)
    args = parser.parse_args()

    assert args.config or args.lsm_param_path
    np.random.seed(args.seed)

    # リキッドステートマシンを作成
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

    if args.lsm_param_path is None:
        lsm_param = network_maker_dict[config["network_maker"]](**config["LSM"])
        lsm = LiquidStateMachine(param=lsm_param)
    else:
        lsm = LiquidStateMachine(load_path=args.lsm_param_path)

    print(f'入力層の結合率    : {lsm.input_connectivity * 100.:.3f}%')
    print(f'リザバー層の結合率: {lsm.rsrvr_connectivity * 100.:.3f}%')
    lsm.plot_eig()

    # ポアソンエンコーダを定義
    encoder = PoissonEncoder(obs_dim=lsm.num_input, max_freq=args.max_input_fire_rate)

    # リードアウト(隠れ層をなしにしたら単純パーセプトロンになる？)
    readout = MLPRegressor(hidden_layer_sizes=())

    # 学習
    spike_log = np.zeros((lsm.num_input, args.pred_frames))   # 直近pred_framesフレームのスパイクを記録する配列
    t_train = []   # 誤差の横軸用
    error_log = []   # 誤差の記録
    start_time = time()
    for step in tqdm(range(int(args.train_time / args.dt))):
        x = np.random.rand(lsm.num_input)
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
    lsm.start_log()
    t_test = []
    true_list = []
    pred_list = []
    spike_log *= 0
    for step in tqdm(range(int(args.test_time / args.dt))):
        x = np.random.rand(lsm.num_input)
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
    lsm.plot_spike_log(args.dt)

    plt.figure()
    plt.plot(t_train, error_log)
    plt.title("Error")

    plt.figure()
    plt.plot(t_test, true_list, label="True")
    plt.plot(t_test, pred_list, label="Pred")
    plt.legend()
    plt.title("Comparison")

    plt.show()
