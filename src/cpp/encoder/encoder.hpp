#pragma once

#include <vector>
#include <random>

class PoissonEncoder
{
  int obs_dim;
  double max_freq;

  std::mt19937 mt;
  std::uniform_real_distribution<double> rand_01;

public:
  PoissonEncoder(int obs_dim, double max_freq);

  std::vector<double> encode(const std::vector<double>& obs, double dt);
};

class PopSANEncoder
{
  int obs_dim;
  int pop_dim;
  int spike_ts;
  double std;
  double v_reset;
  double v_th;

  int encoder_neuron_dim;
  std::vector<double> mean;

public:
  PopSANEncoder(
    int obs_dim,
    int pop_dim,
    int spike_ts,
    double std,
    double v_reset,
    double v_th,
    const std::pair<double, double>& mean_range);

  std::vector<std::vector<double>> encode(const std::vector<double>& obs);
};

class LIFEncoder
{
  int obs_dim;
  int pop_dim;
  double std;
  double v_rest;
  double v_reset;
  double v_th;
  double tau_m;
  double amp;

  int encoder_neuron_dim;
  std::vector<double> mean, pop_volt, pop_spikes;

public:
  LIFEncoder(
    int obs_dim,
    int pop_dim,
    const std::pair<double, double>& mean_range,
    double std,
    double v_rest,
    double v_reset,
    double v_th,
    double tau_m,
    double amp);

  std::vector<double> encode(const std::vector<double>& obs, double dt);

  void reset(const std::vector<double>& init_v);
};
