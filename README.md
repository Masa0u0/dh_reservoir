# dh_snnkit


## Install
---
```bash
$ pip install pybind11
$ cd dh_reservoir
$ pip install -e .
```


## LSMの作り方(ESNもほぼ同じ)
---
- example/に移動する
```bash
$ cd .../dh_snnkit/example/liquid_state_machine
```

- 下記のようなconfigを作成する
```yaml
network_maker: "klampfl2013"

Encoder:
    obs_dim: 8
    pop_dim: 10
    mean_range: [-1., 1.]
    std: 0.1
    amp: 5.5

LSM:
    num_input: 80
    num_wta: 200
    wta_neurons_range: [2, 6]
    k: 2.
    lam: 0.005
    i_noise_scale: 0.
    tau_decay_range: [0.1, 0.1]
    a_distribution: "sc_lognormal"
    a_std_coef: 10.
```

- テストファイルを実行する(下は一例)
```bash
$ python test_periodity.py --config hoge.yaml --save_dir fuga/
```
