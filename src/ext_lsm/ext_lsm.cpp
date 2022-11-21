#include <random>
#include <cassert>

#include "./ext_lsm.hpp"
#include "./wta_circuit.hpp"

using namespace std;

LiquidStateMachine::LiquidStateMachine(
  const vector<vector<float>>& pos_input,
  const vector<vector<float>>& pos_wta,
  const NeuronParams& neuron_params,
  const SynapseParams& synapse_params,
  const pair<int, int>& wta_neurons_range,
  float connectivity,
  float lambda_input,
  float lambda_rsrvr,
  bool stochastic,
  float max_fire_rate,  // stochasticがfalseならば関係ない
  int seed)
{
  assert(pos_input.size() >= 1 && pos_wta.size() >= 1);
  assert(pos_input[0].size() == 3 && pos_wta[0].size() == 3);
  assert(wta_neurons_range.first <= wta_neurons_range.second);
  assert(wta_neurons_range.first > 0);
  assert(lambda_input > 0. && lambda_rsrvr > 0.);

  if (seed < 0)
  {
    random_device rnd;
    seed = rnd();
  }
  mt19937 mt(seed);
  uniform_int_distribution<> seed_gen(0);
  uniform_real_distribution<float> rand_01(0., 1.);
  uniform_int_distribution<> wta_neurons_gen(wta_neurons_range.first, wta_neurons_range.second);

  num_input_neurons = pos_input.size();
  num_rsrvr_neurons = 0;
  num_wta_circuits = pos_wta.size();

  // ニューロンとWTA回路の動的メモリ確保
  for (int i = 0; i < num_input_neurons; i++)
  {
    PseudoNeuron* input_neuron = new PseudoNeuron(neuron_params);
    input_neurons.push_back(input_neuron);
  }
  for (int i = 0; i < num_wta_circuits; i++)
  {
    int wta_neurons = wta_neurons_gen(mt);
    num_rsrvr_neurons += wta_neurons;
    WTACircuit* wta =
      new WTACircuit(wta_neurons, max_fire_rate, stochastic, neuron_params, seed_gen(mt));
    wta_circuits.push_back(wta);
  }

  // 入力層からWTAへのシナプス結合を作る
  for (int i = 0; i < num_input_neurons; i++)
  {
    auto input_neuron = input_neurons[i];
    for (int j = 0; j < num_wta_circuits; j++)
    {
      float dist = sqrt(
        pow((pos_input[i][0] - pos_wta[j][0]), 2.) + pow((pos_input[i][1] - pos_wta[j][1]), 2.)
        + pow((pos_input[i][2] - pos_wta[j][2]), 2.));
      float prob = connectivity * exp(-pow(dist / lambda_input, 2.));
      for (auto rsrvr_neuron : wta_circuits[j]->neurons)
      {
        if (rand_01(mt) < prob)
        {
          DynamicSynapse* synapse = new DynamicSynapse(synapse_params, seed_gen(mt));
          synapse->pre_neuron = input_neuron;
          synapse->post_neuron = rsrvr_neuron;
          rsrvr_neuron->pre_neurons.push_back(input_neuron);
          rsrvr_neuron->pre_synapses.push_back(synapse);
          input_synapses.push_back(synapse);
        }
      }
    }
  }

  // 異なるWTA間のシナプス結合をを作る
  for (int i = 0; i < num_wta_circuits; i++)
  {
    for (int j = 0; j < num_wta_circuits; j++)
    {
      if (i == j)
      {
        continue;
      }
      float dist = sqrt(
        pow((pos_wta[i][0] - pos_wta[j][0]), 2.) + pow((pos_wta[i][1] - pos_wta[j][1]), 2.)
        + pow((pos_wta[i][2] - pos_wta[j][2]), 2.));
      float prob = connectivity * exp(-pow(dist / lambda_rsrvr, 2.));
      for (auto neuron1 : wta_circuits[i]->neurons)
      {
        for (auto neuron2 : wta_circuits[j]->neurons)
        {
          if (rand_01(mt) < prob)
          {
            DynamicSynapse* synapse = new DynamicSynapse(synapse_params, seed_gen(mt));
            synapse->pre_neuron = neuron1;
            synapse->post_neuron = neuron2;
            neuron2->pre_neurons.push_back(neuron1);
            neuron2->pre_synapses.push_back(synapse);
            rsrvr_synapses.push_back(synapse);
          }
        }
      }
    }
  }

  reset();
}

LiquidStateMachine::~LiquidStateMachine()
{
  for (auto neuron : input_neurons)
  {
    delete neuron;
  }
  for (auto synapse : input_synapses)
  {
    delete synapse;
  }
  for (auto synapse : rsrvr_synapses)
  {
    delete synapse;
  }
  for (auto wta : wta_circuits)
  {
    delete wta;
  }
}

void LiquidStateMachine::reset()
{
  for (auto neuron : input_neurons)
  {
    neuron->reset();
  }
  for (auto synapse : input_synapses)
  {
    synapse->reset();
  }
  for (auto synapse : rsrvr_synapses)
  {
    synapse->reset();
  }
  for (auto wta : wta_circuits)
  {
    wta->reset();
  }
}

vector<float> LiquidStateMachine::step(const vector<bool>& input_spikes, float dt)
{
  assert(static_cast<int>(input_spikes.size()) == num_input_neurons);

  // 入力ニューロンにスパイクをセットする
  for (int i = 0; i < num_input_neurons; i++)
  {
    input_neurons[i]->set_spike(input_spikes[i]);
  }

  // 各内部変数の勾配を求める
  for (auto neuron : input_neurons)
  {
    neuron->calc_grad(dt);
  }
  for (auto synapse : input_synapses)
  {
    synapse->calc_grad(dt);
  }
  for (auto synapse : rsrvr_synapses)
  {
    synapse->calc_grad(dt);
  }
  for (auto wta : wta_circuits)
  {
    wta->calc_grad(dt);
  }

  // 各内部変数を更新する
  for (auto neuron : input_neurons)
  {
    neuron->update();
  }
  for (auto synapse : input_synapses)
  {
    synapse->update();
  }
  for (auto synapse : rsrvr_synapses)
  {
    synapse->update();
  }
  for (auto wta : wta_circuits)
  {
    wta->update();
  }

  // ニューロンの発火の処理
  for (auto wta : wta_circuits)
  {
    wta->fire(dt);
  }

  // リザバーニューロンのトレースを取得する
  vector<float> rsrvr_traces;
  for (auto wta : wta_circuits)
  {
    for (auto neuron : wta->neurons)
    {
      rsrvr_traces.push_back(neuron->get_trace());
    }
  }

  return rsrvr_traces;
}

int LiquidStateMachine::get_num_input_neurons()
{
  return num_input_neurons;
}

int LiquidStateMachine::get_num_rsrvr_neurons()
{
  return num_rsrvr_neurons;
}

float LiquidStateMachine::input_connectivity()
{
  return (float)input_synapses.size() / (float)(num_input_neurons * num_rsrvr_neurons);
}

float LiquidStateMachine::rsrvr_connectivity()
{
  return (float)rsrvr_synapses.size() / (float)(num_rsrvr_neurons * num_rsrvr_neurons);
}
