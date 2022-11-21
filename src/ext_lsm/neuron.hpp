#pragma once

#include <vector>

#include "./synapse.hpp"

class DynamicSynapse;

struct NeuronParams
{
public:
  double tau_decay;
  double c;

  NeuronParams();
};

class BaseNeuron
{
protected:
  // const
  double tau_decay;

  // mutable
  bool spike;
  double trace, dtrace;

public:
  virtual ~BaseNeuron()
  {
  }

  virtual void reset()
  {
  }

  virtual void update()
  {
  }

  virtual void calc_grad(double dt)
  {
  }

  void set_spike(bool spike);

  double get_spike();

  double get_trace();
};

class PseudoNeuron : public BaseNeuron
{
public:
  PseudoNeuron(const NeuronParams& params);

  void reset() override;

  void calc_grad(double dt) override;

  void update() override;
};

class LIFNeuron : public BaseNeuron
{
  // const
  double c;

  // mutable
  double w0, dw0, mempot;

public:
  std::vector<BaseNeuron*> pre_neurons;
  std::vector<DynamicSynapse*> pre_synapses;

  LIFNeuron(const NeuronParams& params);

  void reset() override;

  void calc_grad(double dt) override;

  void update() override;

  double get_mempot();
};
