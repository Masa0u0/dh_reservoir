#pragma once

#include <vector>

#include "./neuron.hpp"
#include "./synapse.hpp"
#include "./wta_circuit.hpp"

class LiquidStateMachine
{
  int num_input_neurons;
  int num_rsrvr_neurons;
  int num_wta_circuits;

  std::vector<PseudoNeuron*> input_neurons;
  std::vector<DynamicSynapse*> input_synapses;
  std::vector<DynamicSynapse*> rsrvr_synapses;
  std::vector<WTACircuit*> wta_circuits;

public:
  explicit LiquidStateMachine(
    const std::vector<std::vector<float>>& pos_input,
    const std::vector<std::vector<float>>& pos_wta,
    const NeuronParams& neuron_params,
    const SynapseParams& synapse_params,
    const std::pair<int, int>& wta_neuron_nums_range,
    float connectivity,
    float lambda_input,
    float lambda_rsrvr,
    bool stochastic,
    float fire_rate_max,
    int seed);

  ~LiquidStateMachine();

  void reset();

  std::vector<float> step(const std::vector<bool>& input_spikes, float dt);

  int get_num_input_neurons();

  int get_num_rsrvr_neurons();

  float input_connectivity();

  float rsrvr_connectivity();
};
