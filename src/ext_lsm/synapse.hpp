#ifndef SYNAPSE_H
#define SYNAPSE_H

using namespace std;

class BaseNeuron;


struct SynapseParams
{
public:

    float u0_mean, tau_d_mean, tau_f_mean;
    float u0_sd, tau_d_sd, tau_f_sd;
    float c;

    SynapseParams();
};


class DynamicSynapse
{
    // const
    float u0, tau_d, tau_f, c;

    // mutable
    float u, x, w, amp;
    float du, dx, dw;

public:

    BaseNeuron* pre_neuron;
    BaseNeuron* post_neuron;

    DynamicSynapse(const SynapseParams& params, int seed);

    void reset();

    void calc_grad(float dt);

    void update();

    float get_amp();
};


#endif
