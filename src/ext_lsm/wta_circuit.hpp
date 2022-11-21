#pragma once

#include <vector>
#include <random>

#include "neuron.hpp"

class WTACircuit
{
  std::mt19937 mt;
  std::uniform_real_distribution<float> rand_01;
  float max_fire_rate;
  bool stochastic;

public:
  std::vector<LIFNeuron*> neurons;

  WTACircuit(
    int num_neurons,
    float max_fire_rate,
    bool stochastic,
    const NeuronParams& params,
    int seed);

  ~WTACircuit();

  void reset();

  void calc_grad(float dt);

  void update();

  void fire(float dt);
};
