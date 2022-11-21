#include <cmath>
#include <cassert>
#include <iostream>

#include "./core.hpp"

using namespace std;

Reservoir::Reservoir(int num_neurons, double tau, const vector<Edge>& edges)
  : num_neurons_(num_neurons),
    tau_(tau),
    edges_(edges),
    x_(num_neurons, 0.),
    dot_(num_neurons, 0.)
{
  assert(num_neurons_ > 0);
  assert(tau_ >= 0.);
}

void Reservoir::reset()
{
  fill(x_.begin(), x_.end(), 0.);
  fill(dot_.begin(), dot_.end(), 0.);
}

const vector<double>& Reservoir::step(const vector<double>& x_in, double dt)
{
  assert(x_in.size() == num_neurons_);
  assert(dt > 0.);

  double alpha = dt / (dt + tau_);

  for (const auto& edge : edges_)
  {
    // cout << edge.from << ", " << edge.to << ", " << edge.weight << endl;
    dot_[edge.to] = x_[edge.from] * edge.weight;
  }

  for (int i = 0; i < num_neurons_; ++i)
  {
    x_[i] = (1. - alpha) * x_[i] + alpha * tanh(dot_[i] + x_in[i]);
  }

  return x_;
}

const vector<double>& Reservoir::getState()
{
  return x_;
}
