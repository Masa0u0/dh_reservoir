# 異なる波形パターンがLSM内で別々に表現されているかどうかを評価する

import os
import os.path as osp
import shutil
import argparse
import json
import yaml
import warnings
import optuna
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from time import time
from rich import print

from dh_function.basic import deep_connect
from dh_function.metrics import cross_entropy_error
from dh_reservoir.encoder import LIFEncoder
from dh_reservoir.liquid_state_machine import LiquidStateMachine
from dh_reservoir.liquid_state_machine.network_maker import ponghiran2019, klampfl2013


global config, db_id, obs_dim, num_rsrvr, num_class, num_iter
global sim_time, start_time, dt, freq_range

network_maker_dict = {
    "ponghiran2019": ponghiran2019,
    "klampfl2013": klampfl2013,
}


def objective(trial: optuna.Trial):
    # ====================ハイパーパラメータの取得====================
    hyper_params = {
        "Encoder": dict(),
        "LSM": dict(),
    }

    # Encoder
    hyper_params["Encoder"]["amp"] = trial.suggest_int("amp", 2, 10)

    # LSM
    hyper_params["LSM"]["k"] = trial.suggest_int("k", 2, 7)
    tau_decay = trial.suggest_float("tau_decay", 0.01, 0.2)
    hyper_params["LSM"]["tau_decay_range"] = (tau_decay, tau_decay)

    # ====================log_dirを作成====================
    log_dir = osp.join(
        osp.dirname(__file__),
        "logs",
        db_id,
        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    )
    os.makedirs(log_dir, exist_ok=True)

    # ====================オブジェクトの作成====================
    encoder = LIFEncoder(**deep_connect(config["Encoder"], hyper_params["Encoder"]))
    lsm_param = network_maker_dict[config["network_maker"]](
        **deep_connect(config["LSM"], hyper_params["LSM"])
    )
    lsm = LiquidStateMachine(param=lsm_param)

    # ====================学習と評価====================
    acc_list = []
    cer_list = []
    for iter in range(num_iter):
        X = []
        y = []

        # データセットの作成
        start_time = time()

        for cls in tqdm(range(num_class)):
            # 入力信号生成用パラメータ
            freq = np.random.uniform(freq_range[0], freq_range[1], (encoder.obs_dim,))
            delta = np.random.uniform(0., 100., (encoder.obs_dim,))

            target = np.zeros((num_class,))
            target[cls] = 1.

            lsm.reset()
            for step in range(sim_steps):
                t = step * dt
                obs = np.sin(2 * np.pi * freq * t + delta)
                input_spike = encoder.encode(obs, dt)
                rsrvr_trace = lsm.step(input_spike, dt)

                if step > start_steps:
                    X.append(rsrvr_trace)
                    y.append(target)

        end_time = time()
        elapsed_time = end_time - start_time
        time_for_1ms_sim = elapsed_time / (num_class * sim_time)

        X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y))
        # print(np.argmax(y_train[:20, :], axis=1))

        # 学習
        model = MLPClassifier(hidden_layer_sizes=())
        model.fit(X_train, y_train)

        # 評価
        y_pred = model.predict_proba(X_test)
        acc = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        cer = cross_entropy_error(y_test, y_pred)
        acc_list.append(acc)
        cer_list.append(cer)

        # 表示
        print(
            f'iter: {iter + 1}, '
            f'accuray: {acc:.3f}, '
            f'cross_entropy_error: {cer:.3f} '
            f'time_for_1ms_sim: {time_for_1ms_sim:.3f}'
        )

    mean_acc = np.mean(acc_list)
    mean_cer = np.mean(cer_list)

    # ====================後処理====================
    # ハイパラと結果とモデルを保存
    with open(osp.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    with open(osp.join(log_dir, "hyper_params.json"), "w") as f:
        json.dump(hyper_params, f, indent=4)
    with open(osp.join(log_dir, "log.txt"), "a") as f:
        f.write(
            f'accuracy             : {mean_acc}\n'
            f'cross_entropy_error  : {mean_cer}\n'
            f'time_for_1ms_sim     : {time_for_1ms_sim}\n'
        )
    encoder.save(osp.join(log_dir, "encoder_params.pkl"))
    lsm.save(osp.join(log_dir, "lsm_params.pkl"))
    shutil.move(log_dir, f'{log_dir}_{mean_cer:.3f}')

    # オブジェクトを削除
    del hyper_params, log_dir, encoder, lsm, acc_list, model
    del X, y, X_train, X_test, y_train, y_test

    return mean_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="設定ファイル")
    parser.add_argument("--db_id", type=str, required=True, help="ユニークならば何でもよい")
    parser.add_argument("--n_trials", type=int, required=True, help="1つのウィンドウでの試行回数")
    parser.add_argument("--num_class", type=int, default=100, help="分類するサイン波のパターン数")
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--sim_time", type=float, default=3.)
    parser.add_argument("--start_time", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=1e-3, help="LSMのタイムステップ")
    parser.add_argument("--freq_range", type=list, default=[0., 10.])

    args = parser.parse_args()

    warnings.resetwarnings()
    warnings.simplefilter("ignore", UserWarning)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    db_id = args.db_id
    num_class = args.num_class
    num_iter = args.num_iter
    sim_time = args.sim_time
    start_time = args.start_time
    dt = args.dt
    freq_range = args.freq_range

    sim_steps = int(sim_time / dt)
    start_steps = int(start_time / dt)

    # ハイパラ最適化
    study = optuna.create_study(
        study_name=db_id,
        storage=f'sqlite:///{db_id}.db',
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(),
    )
    study.optimize(objective, n_trials=args.n_trials)
