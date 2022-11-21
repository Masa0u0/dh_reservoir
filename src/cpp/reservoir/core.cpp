#include <cmath>

#include "./core.hpp"

using namespace std;

Reservoir::Reservoir(
  int num_neurons,
  double tau,
  const vector<pair<int, int>>& pairs,
  const vector<double>& weights)
  : num_neurons{ num_neurons }, tau{ tau }, pairs{ pairs }, weights{ weights }
{
  reset();
}

void Reservoir::reset()
{
  x = vector<double>(num_neurons, 0.);
  dot = vector<double>(num_neurons, 0.);
}

vector<double> Reservoir::step(const vector<double>& x_in, double dt)
{
  double alpha = dt / (dt + tau);

  fill(dot.begin(), dot.end(), 0.);
  for (int i = 0; i < pairs.size(); i++)
  {
    dot[pairs[i].second] = x[pairs[i].first] * weights[i];
  }

  for (int i = 0; i < num_neurons; i++)
  {
    x[i] = (1. - alpha) * x[i] + alpha * tanh(dot[i] + x_in[i]);
  }

  return x;
}

vector<double> Reservoir::get_state()
{
  return x;
}
