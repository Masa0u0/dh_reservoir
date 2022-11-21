#pragma once

#include "./neuron.hpp"

class BaseNeuron;

struct SynapseParams
{
public:
  double u0_mean, tau_d_mean, tau_f_mean;
  double u0_sd, tau_d_sd, tau_f_sd;
  double c;

  SynapseParams();
};

class DynamicSynapse
{
  // const
  double u0, tau_d, tau_f, c;

  // mutable
  double u, x, w, amp;
  double du, dx, dw;

public:
  BaseNeuron* pre_neuron;
  BaseNeuron* post_neuron;

  DynamicSynapse(const SynapseParams& params, int seed);

  void reset();

  void calc_grad(double dt);

  void update();

  double get_amp();
};
