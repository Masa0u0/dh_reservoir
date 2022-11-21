#pragma once

#include <vector>
#include <random>

struct LSMParam
{
  int num_input;
  int num_rsrvr;
  int num_syn_input;
  int num_syn_rsrvr;
  double tau_m;
  double v_th;
  double v_rest;
  double v_reset;
  double i_back;
  double i_noise_scale;
  double r;
  std::vector<bool> is_exc_input;
  std::vector<bool> is_exc_rsrvr;
  std::vector<std::pair<int, int>> pairs_input;
  std::vector<std::pair<int, int>> pairs_rsrvr;
  std::vector<double> t_ref;
  std::vector<double> tau_decay;
  std::vector<double> tau_syn_input;
  std::vector<double> tau_syn_rsrvr;
  std::vector<double> tau_d;
  std::vector<double> tau_f;
  std::vector<double> u0;
  std::vector<double> a_input;
  std::vector<double> a_rsrvr;
  std::vector<double> delay_input;
  std::vector<double> delay_rsrvr;

  LSMParam(
    int num_input,
    int num_rsrvr,
    int num_syn_input,
    int num_syn_rsrvr,
    double tau_m,
    double v_th,
    double v_rest,
    double v_reset,
    double i_back,
    double i_noise_scale,
    double r,
    const std::vector<bool>& is_exc_input,
    const std::vector<bool>& is_exc_rsrvr,
    const std::vector<std::pair<int, int>>& pairs_input,
    const std::vector<std::pair<int, int>>& pairs_rsrvr,
    const std::vector<double>& t_ref,
    const std::vector<double>& tau_decay,
    const std::vector<double>& tau_syn_input,
    const std::vector<double>& tau_syn_rsrvr,
    const std::vector<double>& tau_d,
    const std::vector<double>& tau_f,
    const std::vector<double>& u0,
    const std::vector<double>& a_input,
    const std::vector<double>& a_rsrvr,
    const std::vector<double>& delay_input,
    const std::vector<double>& delay_rsrvr);
};

class LiquidStateMachine
{
  LSMParam param;
  int n_step;
  bool log_flg;
  std::vector<double> u, x, i_input, i_input_delayed, i_rsrvr, i_rsrvr_delayed, i_sum;
  std::vector<double> s_re, v, trace, t_last, di_input, di_rsrvr, du, dx, dv, dtrace;
  std::vector<std::vector<int>> spike_log;
  std::mt19937 mt;
  std::uniform_real_distribution<double> rand1;

public:
  LiquidStateMachine(const LSMParam& param, int seed);

  void reset(const std::vector<double>& init_v);

  std::vector<double> step(const std::vector<double>& s_in, double dt);

  void start_log();

  std::vector<double> get_trace();

  std::vector<std::vector<int>> get_spike_log();
};
