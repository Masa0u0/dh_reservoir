#include <cmath>

#include "./neuron.hpp"

using namespace std;

// NeuronParams

NeuronParams::NeuronParams()
{
  tau_decay = 0.02;
  c = 0.05 * exp(5.);
}

// BaseNeuron

void BaseNeuron::set_spike(bool spike)
{
  this->spike = spike;
  if (spike)
  {
    // trace = 1.;
    trace += 1.;
  }
}

float BaseNeuron::get_spike()
{
  return static_cast<float>(spike);
}

float BaseNeuron::get_trace()
{
  return trace;
}

// PseudoNeuron

PseudoNeuron::PseudoNeuron(const NeuronParams& params)
{
  tau_decay = params.tau_decay;
}

void PseudoNeuron::reset()
{
  spike = false;
  trace = 0.;
  dtrace = 0.;
}

void PseudoNeuron::calc_grad(float dt)
{
  dtrace = -(dt / tau_decay) * trace;
}

void PseudoNeuron::update()
{
  trace += dtrace;
}

// LIFNeuron

LIFNeuron::LIFNeuron(const NeuronParams& params)
{
  tau_decay = params.tau_decay;
  c = params.c;
}

void LIFNeuron::reset()
{
  spike = false;
  trace = 0.;
  dtrace = 0.;
  w0 = 0.;
  dw0 = 0.;
  mempot = 0.;
}

void LIFNeuron::calc_grad(float dt)
{
  dtrace = -(dt / tau_decay) * trace;
  dw0 = (c * exp(-w0) * get_spike() - 1.) * dt;
}

void LIFNeuron::update()
{
  trace += dtrace;
  w0 += dw0;

  mempot = w0;
  for (int i = 0; i < static_cast<int>(pre_neurons.size()); i++)
  {
    mempot += pre_neurons[i]->get_trace() * pre_synapses[i]->get_amp();
  }

  // cout << "trace: " << trace << ", w0: " << w0 << ", mempot: " << mempot << endl;
}

float LIFNeuron::get_mempot()
{
  return mempot;
}
