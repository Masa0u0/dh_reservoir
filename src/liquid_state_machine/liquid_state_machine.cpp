#include "./liquid_state_machine.hpp"

using namespace std;

LSMParam::LSMParam(
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
  const vector<bool>& is_exc_input,
  const vector<bool>& is_exc_rsrvr,
  const vector<pair<int, int>>& pairs_input,
  const vector<pair<int, int>>& pairs_rsrvr,
  const vector<double>& t_ref,
  const vector<double>& tau_decay,
  const vector<double>& tau_syn_input,
  const vector<double>& tau_syn_rsrvr,
  const vector<double>& tau_d,
  const vector<double>& tau_f,
  const vector<double>& u0,
  const vector<double>& a_input,
  const vector<double>& a_rsrvr,
  const vector<double>& delay_input,
  const vector<double>& delay_rsrvr)
  : num_input{ num_input },
    num_rsrvr{ num_rsrvr },
    num_syn_input{ num_syn_input },
    num_syn_rsrvr{ num_syn_rsrvr },
    tau_m{ tau_m },
    v_th{ v_th },
    v_rest{ v_rest },
    v_reset{ v_reset },
    i_back{ i_back },
    i_noise_scale{ i_noise_scale },
    r{ r },
    is_exc_input{ is_exc_input },
    is_exc_rsrvr{ is_exc_rsrvr },
    pairs_input{ pairs_input },
    pairs_rsrvr{ pairs_rsrvr },
    t_ref{ t_ref },
    tau_decay{ tau_decay },
    tau_syn_input{ tau_syn_input },
    tau_syn_rsrvr{ tau_syn_rsrvr },
    tau_d{ tau_d },
    tau_f{ tau_f },
    u0{ u0 },
    a_input{ a_input },
    a_rsrvr{ a_rsrvr },
    delay_input{ delay_input },
    delay_rsrvr{ delay_rsrvr }
{
}

LiquidStateMachine::LiquidStateMachine(const LSMParam& param, int seed = -1) : param{ param }
{
  if (seed < 0)
  {
    random_device rnd;
    seed = rnd();
  }
  mt = mt19937(seed);
  rand1 = uniform_real_distribution<double>(-1., 1.);
}

void LiquidStateMachine::reset(const vector<double>& init_v)
{
  v = init_v;
  n_step = 0;
  u = vector<double>(param.num_syn_rsrvr, 0.);
  x = vector<double>(param.num_syn_rsrvr, 0.);
  i_input = vector<double>(param.num_syn_input, 0.);
  i_input_delayed = vector<double>(param.num_syn_input, 0.);
  i_rsrvr = vector<double>(param.num_syn_rsrvr, 0.);
  i_rsrvr_delayed = vector<double>(param.num_syn_rsrvr, 0.);
  i_sum = vector<double>(param.num_rsrvr, 0.);
  trace = vector<double>(param.num_rsrvr, 0.);
  s_re = vector<double>(param.num_rsrvr, 0.);
  t_last = vector<double>(param.num_rsrvr, -HUGE_VALF);
  di_input = vector<double>(param.num_syn_input, 0.);
  di_rsrvr = vector<double>(param.num_syn_rsrvr, 0.);
  du = vector<double>(param.num_syn_rsrvr, 0.);
  dx = vector<double>(param.num_syn_rsrvr, 0.);
  dv = vector<double>(param.num_rsrvr, 0.);
  dtrace = vector<double>(param.num_rsrvr, 0.);
  log_flg = false;
  spike_log.clear();
}

vector<double> LiquidStateMachine::step(const vector<double>& s_in, double dt)
{
  int i;
  double delta, alpha;

  // ステップを進める
  n_step += 1;

  // 入力層のシナプス後電流のdt間での変化量を計算
  for (i = 0; i < param.num_syn_input; i++)
  {
    delta = s_in[param.pairs_input[i].first];
    di_input[i] = (-dt / param.tau_syn_input[i]) * i_input[i] + param.a_input[i] * delta;
  }

  // リザバー層のシナプス後電流のdt間での変化量を計算
  for (i = 0; i < param.num_syn_rsrvr; i++)
  {
    delta = s_re[param.pairs_rsrvr[i].first];
    du[i] = (-dt / param.tau_f[i]) * u[i] + param.u0[i] * (1. - u[i]) * delta;
    dx[i] = (dt / param.tau_d[i]) * (1. - x[i]) - u[i] * x[i] * delta;
    di_rsrvr[i] =
      ((-dt / param.tau_syn_rsrvr[i]) * i_rsrvr[i] + param.a_rsrvr[i] * u[i] * x[i] * delta);
  }

  // 遅延した各シナプス後電流から、各ニューロンに入る電流を計算する
  fill(i_sum.begin(), i_sum.end(), 0.);
  for (i = 0; i < param.num_syn_input; i++)
  {
    i_sum[param.pairs_input[i].second] += i_input_delayed[i];
  }
  for (i = 0; i < param.num_syn_rsrvr; i++)
  {
    i_sum[param.pairs_rsrvr[i].second] += i_rsrvr_delayed[i];
  }

  // 各ニューロンの膜電位とトレースのdt間での変化量を計算
  for (i = 0; i < param.num_rsrvr; i++)
  {
    if (dt * n_step > t_last[i] + param.t_ref[i])
    {
      double i_noise = param.i_noise_scale * rand1(mt);
      dv[i] =
        (param.v_rest - v[i] + param.r * (i_sum[i] + param.i_back + i_noise)) * dt / param.tau_m;
    }
    else
    {
      dv[i] = 0.;
    }

    dtrace[i] = -(dt / param.tau_decay[i]) * trace[i];
  }

  // 各内部変数を更新
  for (i = 0; i < param.num_syn_input; i++)
  {
    i_input[i] += di_input[i];
    alpha = dt / (param.delay_input[i] + dt);
    i_input_delayed[i] = alpha * i_input[i] + (1. - alpha) * i_input_delayed[i];
  }
  for (i = 0; i < param.num_syn_rsrvr; i++)
  {
    u[i] += du[i];
    x[i] += dx[i];
    i_rsrvr[i] += di_rsrvr[i];
    alpha = dt / (param.delay_rsrvr[i] + dt);
    i_rsrvr_delayed[i] = alpha * i_rsrvr[i] + (1. - alpha) * i_rsrvr_delayed[i];
  }

  vector<int> spike_idx;
  for (i = 0; i < param.num_rsrvr; i++)
  {
    v[i] += dv[i];
    trace[i] += dtrace[i];

    // 閾値を超えていたら発火
    if (v[i] > param.v_th)
    {
      spike_idx.push_back(i);
      s_re[i] = 1.;
      // trace[i] = 1.;
      trace[i] += 1;
      t_last[i] = dt * n_step;
      v[i] = param.v_reset;
    }
    else
    {
      s_re[i] = 0.;
    }
  }
  spike_idx.push_back(-1);  // spike_idxのサイズが0になるのを防ぐ
  spike_log.push_back(spike_idx);

  // return s_re;
  return trace;
}

void LiquidStateMachine::start_log()
{
  log_flg = true;
}

vector<double> LiquidStateMachine::get_trace()
{
  return trace;
}

vector<vector<int>> LiquidStateMachine::get_spike_log()
{
  return spike_log;
}
