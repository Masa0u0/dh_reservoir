#pragma once

#include <vector>

class Reservoir
{
  int num_neurons;
  float tau;
  std::vector<std::pair<int, int>> pairs;
  std::vector<float> weights, x, dot;

public:
  Reservoir(
    int num_neurons,
    float tau,
    const std::vector<std::pair<int, int>>& pairs,
    const std::vector<float>& weights);

  void reset();

  std::vector<float> step(const std::vector<float>& x_in, float dt);

  std::vector<float> get_state();
};
