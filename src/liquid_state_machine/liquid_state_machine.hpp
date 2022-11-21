#pragma once

#include <vector>
#include <random>

struct LSMParam
{
  int num_input;
  int num_rsrvr;
  int num_syn_input;
  int num_syn_rsrvr;
  float tau_m;
  float v_th;
  float v_rest;
  float v_reset;
  float i_back;
  float i_noise_scale;
  float r;
  std::vector<bool> is_exc_input;
  std::vector<bool> is_exc_rsrvr;
  std::vector<std::pair<int, int>> pairs_input;
  std::vector<std::pair<int, int>> pairs_rsrvr;
  std::vector<float> t_ref;
  std::vector<float> tau_decay;
  std::vector<float> tau_syn_input;
  std::vector<float> tau_syn_rsrvr;
  std::vector<float> tau_d;
  std::vector<float> tau_f;
  std::vector<float> u0;
  std::vector<float> a_input;
  std::vector<float> a_rsrvr;
  std::vector<float> delay_input;
  std::vector<float> delay_rsrvr;

  LSMParam(
    int num_input,
    int num_rsrvr,
    int num_syn_input,
    int num_syn_rsrvr,
    float tau_m,
    float v_th,
    float v_rest,
    float v_reset,
    float i_back,
    float i_noise_scale,
    float r,
    const std::vector<bool>& is_exc_input,
    const std::vector<bool>& is_exc_rsrvr,
    const std::vector<std::pair<int, int>>& pairs_input,
    const std::vector<std::pair<int, int>>& pairs_rsrvr,
    const std::vector<float>& t_ref,
    const std::vector<float>& tau_decay,
    const std::vector<float>& tau_syn_input,
    const std::vector<float>& tau_syn_rsrvr,
    const std::vector<float>& tau_d,
    const std::vector<float>& tau_f,
    const std::vector<float>& u0,
    const std::vector<float>& a_input,
    const std::vector<float>& a_rsrvr,
    const std::vector<float>& delay_input,
    const std::vector<float>& delay_rsrvr);
};

class LiquidStateMachine
{
  LSMParam param;
  int n_step;
  bool log_flg;
  std::vector<float> u, x, i_input, i_input_delayed, i_rsrvr, i_rsrvr_delayed, i_sum;
  std::vector<float> s_re, v, trace, t_last, di_input, di_rsrvr, du, dx, dv, dtrace;
  std::vector<std::vector<int>> spike_log;
  std::mt19937 mt;
  std::uniform_real_distribution<float> rand1;

public:
  LiquidStateMachine(const LSMParam& param, int seed);

  void reset(const std::vector<float>& init_v);

  std::vector<float> step(const std::vector<float>& s_in, float dt);

  void start_log();

  std::vector<float> get_trace();

  std::vector<std::vector<int>> get_spike_log();
};
