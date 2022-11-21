#ifndef EXT_LSM_H
#define EXT_LSM_H

using namespace std;


class LiquidStateMachine
{
    int num_input_neurons;
    int num_rsrvr_neurons;
    int num_wta_circuits;

    vector<PseudoNeuron*> input_neurons;
    vector<DynamicSynapse*> input_synapses;
    vector<DynamicSynapse*> rsrvr_synapses;
    vector<WTACircuit*> wta_circuits;

public:

    explicit LiquidStateMachine(
        const vector<vector<float>>& pos_input,
        const vector<vector<float>>& pos_wta,
        const NeuronParams& neuron_params,
        const SynapseParams& synapse_params,
        const pair<int, int>& wta_neuron_nums_range,
        float connectivity,
        float lambda_input,
        float lambda_rsrvr,
        bool stochastic,
        float fire_rate_max,
        int seed
    );

    ~LiquidStateMachine();

    void reset();

    vector<float> step(const vector<bool>& input_spikes, float dt);

    int get_num_input_neurons();

    int get_num_rsrvr_neurons();

    float input_connectivity();

    float rsrvr_connectivity();
};


#endif
