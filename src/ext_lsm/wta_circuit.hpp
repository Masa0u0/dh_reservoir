#pragma once

#include <vector>
#include <random>

#include "neuron.hpp"

class WTACircuit
{
  std::mt19937 mt;
  std::uniform_real_distribution<double> rand_01;
  double max_fire_rate;
  bool stochastic;

public:
  std::vector<LIFNeuron*> neurons;

  WTACircuit(
    int num_neurons,
    double max_fire_rate,
    bool stochastic,
    const NeuronParams& params,
    int seed);

  ~WTACircuit();

  void reset();

  void calc_grad(double dt);

  void update();

  void fire(double dt);
};
