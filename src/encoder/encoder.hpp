#pragma once

#include <vector>
#include <random>

class PoissonEncoder
{
  int obs_dim;
  float max_freq;

  std::mt19937 mt;
  std::uniform_real_distribution<float> rand_01;

public:
  PoissonEncoder(int obs_dim, float max_freq);

  std::vector<float> encode(const std::vector<float>& obs, float dt);
};

class PopSANEncoder
{
  int obs_dim;
  int pop_dim;
  int spike_ts;
  float std;
  float v_reset;
  float v_th;

  int encoder_neuron_dim;
  std::vector<float> mean;

public:
  PopSANEncoder(
    int obs_dim,
    int pop_dim,
    int spike_ts,
    float std,
    float v_reset,
    float v_th,
    const std::pair<float, float>& mean_range);

  std::vector<std::vector<float>> encode(const std::vector<float>& obs);
};

class LIFEncoder
{
  int obs_dim;
  int pop_dim;
  float std;
  float v_rest;
  float v_reset;
  float v_th;
  float tau_m;
  float amp;

  int encoder_neuron_dim;
  std::vector<float> mean, pop_volt, pop_spikes;

public:
  LIFEncoder(
    int obs_dim,
    int pop_dim,
    const std::pair<float, float>& mean_range,
    float std,
    float v_rest,
    float v_reset,
    float v_th,
    float tau_m,
    float amp);

  std::vector<float> encode(const std::vector<float>& obs, float dt);

  void reset(const std::vector<float>& init_v);
};
