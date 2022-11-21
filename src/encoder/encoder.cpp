#include <cassert>
#include <stdexcept>

#include "./encoder.hpp"
#include "./util.hpp"

using namespace std;

PoissonEncoder::PoissonEncoder(int obs_dim, float max_freq)
  : obs_dim{ obs_dim }, max_freq{ max_freq }
{
  assert(obs_dim >= 1);
  assert(max_freq > 0.);

  random_device rnd;
  mt = mt19937(rnd());
  rand_01 = uniform_real_distribution<float>(0., 1.);
}

vector<float> PoissonEncoder::encode(const vector<float>& obs, float dt)
{
  assert(static_cast<int>(obs.size()) == obs_dim);
  float c = max_freq * dt;
  vector<float> res(obs_dim, 0.);
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
  float std,
  float v_reset,
  float v_th,
  const pair<float, float>& mean_range)
  : obs_dim{ obs_dim },
    pop_dim{ pop_dim },
    spike_ts{ spike_ts },
    std{ std },
    v_reset{ v_reset },
    v_th{ v_th }
{
  encoder_neuron_dim = obs_dim * pop_dim;
  mean = vector<float>(pop_dim);
  float delta_mean = (mean_range.second - mean_range.first) / (float)(pop_dim - 1);
  for (int i = 0; i < pop_dim; i++)
  {
    mean[i] = mean_range.first + delta_mean * i;
  }
}

vector<vector<float>> PopSANEncoder::encode(const vector<float>& obs)
{
  vector<float> pop_act(encoder_neuron_dim);
  for (int i = 0; i < obs_dim; i++)
  {
    for (int j = 0; j < pop_dim; j++)
    {
      pop_act[pop_dim * i + j] = rbf_kernel(obs[i], mean[j], std);
    }
  }

  vector<float> pop_volt(encoder_neuron_dim, 0.);
  vector<vector<float>> pop_spikes(encoder_neuron_dim, vector<float>(spike_ts, 0.));

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
  const pair<float, float>& mean_range,
  float std,
  float v_rest,
  float v_reset,
  float v_th,
  float tau_m,
  float amp)
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
  mean = vector<float>(pop_dim);
  float delta_mean = (mean_range.second - mean_range.first) / (float)(pop_dim - 1);
  for (int i = 0; i < pop_dim; i++)
  {
    mean[i] = mean_range.first + delta_mean * i;
  }
}

vector<float> LIFEncoder::encode(const vector<float>& obs, float dt)
{
  if (dt >= tau_m)
  {
    throw invalid_argument("dt must be lower than the membrane time constant");
  }

  vector<float> pop_act(encoder_neuron_dim);
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

void LIFEncoder::reset(const vector<float>& init_v)
{
  pop_volt = init_v;
  pop_spikes = vector<float>(encoder_neuron_dim, 0.);
}
