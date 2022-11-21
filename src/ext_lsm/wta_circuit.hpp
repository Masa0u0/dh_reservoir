#ifndef WTA_CIRCUIT_H
#define WTA_CIRCUIT_H

using namespace std;


class WTACircuit
{
    mt19937 mt;
    uniform_real_distribution<float> rand_01;
    float max_fire_rate;
    bool stochastic;

public:

    vector<LIFNeuron*> neurons;

    WTACircuit(
        int num_neurons,
        float max_fire_rate,
        bool stochastic,
        const NeuronParams& params,
        int seed
    );

    ~WTACircuit();

    void reset();

    void calc_grad(float dt);

    void update();

    void fire(float dt);
};


#endif
