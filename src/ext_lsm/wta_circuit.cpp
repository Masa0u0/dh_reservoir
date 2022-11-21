#include "./wta_circuit.hpp"

using namespace std;

WTACircuit::WTACircuit(
  int num_neurons,
  double max_fire_rate,
  bool stochastic,
  const NeuronParams& params,
  int seed)
  : max_fire_rate{ max_fire_rate }, stochastic{ stochastic }
{
  mt = mt19937(seed);
  rand_01 = uniform_real_distribution<double>(0., 1.);

  for (int i = 0; i < num_neurons; i++)
  {
    LIFNeuron* neuron = new LIFNeuron(params);
    this->neurons.push_back(neuron);
  }
}

WTACircuit::~WTACircuit()
{
  for (auto neuron : neurons)
  {
    delete neuron;
  }
}

void WTACircuit::reset()
{
  for (auto neuron : neurons)
  {
    neuron->reset();
  }
}

void WTACircuit::calc_grad(double dt)
{
  for (auto neuron : neurons)
  {
    neuron->calc_grad(dt);
  }
}

void WTACircuit::update()
{
  for (auto neuron : neurons)
  {
    neuron->update();
  }
}

void WTACircuit::fire(double dt)
{
  if (stochastic)
  {
    double sum_exp = 0.;
    for (auto neuron : neurons)
    {
      sum_exp += exp(neuron->get_mempot());
    }

    for (auto neuron : neurons)
    {
      double fire_prob = dt * max_fire_rate * exp(neuron->get_mempot()) / sum_exp;
      if (rand_01(mt) < fire_prob)
      {
        neuron->set_spike(true);
      }
    }
  }
  else
  {
    // 確率的に発火させない場合、発火率がdtやWTA内のニューロン数に依存することに注意
    int idx;
    double max_mempot = -HUGE_VALF;
    for (int i = 0; i < neurons.size(); i++)
    {
      if (neurons[i]->get_mempot() > max_mempot)
      {
        idx = i;
        max_mempot = neurons[i]->get_mempot();
      }
    }
    neurons[idx]->set_spike(true);
  }
}
