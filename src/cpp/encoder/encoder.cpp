#include <cassert>
#include <stdexcept>

#include "./encoder.hpp"
#include "./util.hpp"

using namespace std;

PoissonEncoder::PoissonEncoder(int obs_dim, double max_freq)
  : obs_dim{ obs_dim }, max_freq{ max_freq }
{
  assert(obs_dim >= 1);
  assert(max_freq > 0.);

  random_device rnd;
  mt = mt19937(rnd());
  rand_01 = uniform_real_distribution<double>(0., 1.);
}

vector<double> PoissonEncoder::encode(const vector<double>& obs, double dt)
{
  assert(static_cast<int>(obs.size()) == obs_dim);
  double c = max_freq * dt;
  vector<double> res(obs_dim, 0.);
  for (int i = 0; i < obs_dim; i++)
  {
    assert(0. <= obs[i] && obs[i] <= 1.);
    if (rand_01(mt) < obs[i] * c)
    {
      res[i] = 1.;
    }
  }
  return res;
}

PopSANEncoder::PopSANEncoder(
  int obs_dim,
  int pop_dim,
  int spike_ts,
  double std,
  double v_reset,
  double v_th,
  const pair<double, double>& mean_range)
  : obs_dim{ obs_dim },
    pop_dim{ pop_dim },
    spike_ts{ spike_ts },
    std{ std },
    v_reset{ v_reset },
    v_th{ v_th }
{
  encoder_neuron_dim = obs_dim * pop_dim;
  mean = vector<double>(pop_dim);
  double delta_mean = (mean_range.second - mean_range.first) / (double)(pop_dim - 1);
  for (int i = 0; i < pop_dim; i++)
  {
    mean[i] = mean_range.first + delta_mean * i;
  }
}

vector<vector<double>> PopSANEncoder::encode(const vector<double>& obs)
{
  vector<double> pop_act(encoder_neuron_dim);
  for (int i = 0; i < obs_dim; i++)
  {
    for (int j = 0; j < pop_dim; j++)
    {
      pop_act[pop_dim * i + j] = rbf_kernel(obs[i], mean[j], std);
    }
  }

  vector<double> pop_volt(encoder_neuron_dim, 0.);
  vector<vector<double>> pop_spikes(encoder_neuron_dim, vector<double>(spike_ts, 0.));

  for (int step = 0; step < spike_ts; step++)
  {
    for (int i = 0; i < encoder_neuron_dim; i++)
    {
      pop_volt[i] += pop_act[i];
      if (pop_volt[i] > v_th)
      {
        pop_spikes[i][step] = 1.;
        pop_volt[i] = v_reset;
      }
    }
  }

  return pop_spikes;
}

LIFEncoder::LIFEncoder(
  int obs_dim,
  int pop_dim,
  const pair<double, double>& mean_range,
  double std,
  double v_rest,
  double v_reset,
  double v_th,
  double tau_m,
  double amp)
  : obs_dim{ obs_dim },
    pop_dim{ pop_dim },
    std{ std },
    v_rest{ v_rest },
    v_reset{ v_reset },
    v_th{ v_th },
    tau_m{ tau_m },
    amp{ amp }
{
  encoder_neuron_dim = obs_dim * pop_dim;
  mean = vector<double>(pop_dim);
  double delta_mean = (mean_range.second - mean_range.first) / (double)(pop_dim - 1);
  for (int i = 0; i < pop_dim; i++)
  {
    mean[i] = mean_range.first + delta_mean * i;
  }
}

vector<double> LIFEncoder::encode(const vector<double>& obs, double dt)
{
  if (dt >= tau_m)
  {
    throw invalid_argument("dt must be lower than the membrane time constant");
  }

  vector<double> pop_act(encoder_neuron_dim);
  for (int i = 0; i < obs_dim; i++)
  {
    for (int j = 0; j < pop_dim; j++)
    {
      pop_act[pop_dim * i + j] = rbf_kernel(obs[i], mean[j], std);
    }
  }

  for (int i = 0; i < encoder_neuron_dim; i++)
  {
    pop_volt[i] += (v_rest + amp * pop_act[i] - pop_volt[i]) * dt / tau_m;
    if (pop_volt[i] > v_th)
    {
      pop_spikes[i] = 1.;
      pop_volt[i] = v_reset;
    }
    else
    {
      pop_spikes[i] = 0.;
    }
  }

  return pop_spikes;
}

void LIFEncoder::reset(const vector<double>& init_v)
{
  pop_volt = init_v;
  pop_spikes = vector<double>(encoder_neuron_dim, 0.);
}
