#pragma once

#include <vector>

class Reservoir
{
  int num_neurons;
  double tau;
  std::vector<std::pair<int, int>> pairs;
  std::vector<double> weights, x, dot;

public:
  Reservoir(
    int num_neurons,
    double tau,
    const std::vector<std::pair<int, int>>& pairs,
    const std::vector<double>& weights);

  void reset();

  std::vector<double> step(const std::vector<double>& x_in, double dt);

  std::vector<double> get_state();
};
