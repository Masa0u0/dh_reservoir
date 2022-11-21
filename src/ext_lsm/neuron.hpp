#ifndef NEURON_H
#define NEURON_H

using namespace std;

class DynamicSynapse;


struct NeuronParams
{
public:

    float tau_decay;
    float c;

    NeuronParams();
};


class BaseNeuron
{
protected:

    // const
    float tau_decay;

    // mutable
    bool spike;
    float trace, dtrace;

public:

    virtual ~BaseNeuron() { }

    virtual void reset() { }

    virtual void update() { }

    virtual void calc_grad(float dt) { }

    void set_spike(bool spike);

    float get_spike();

    float get_trace();

};


class PseudoNeuron : public BaseNeuron
{
public:

    PseudoNeuron(const NeuronParams& params);

    void reset() override;

    void calc_grad(float dt) override;

    void update() override;
};


class LIFNeuron : public BaseNeuron
{
    // const
    float c;

    // mutable
    float w0, dw0, mempot;

public:

    vector<BaseNeuron*> pre_neurons;
    vector<DynamicSynapse*> pre_synapses;

    LIFNeuron(const NeuronParams& params);

    void reset() override;

    void calc_grad(float dt) override;

    void update() override;

    float get_mempot();
};


#endif
