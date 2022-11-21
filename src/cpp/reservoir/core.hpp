#pragma once

#include <vector>

struct Edge
{
  int from;
  int to;
  double weight;
};

class Reservoir
{
public:
  Reservoir(int num_neurons, double tau, const std::vector<Edge>& edges);

  void reset();

  const std::vector<double>& step(const std::vector<double>& x_in, double dt);

  const std::vector<double>& getState();

private:
  const int num_neurons_;
  const double tau_;
  const std::vector<Edge> edges_;
  std::vector<double> x_;
  std::vector<double> dot_;
};
