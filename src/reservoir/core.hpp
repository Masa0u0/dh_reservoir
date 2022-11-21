#ifndef RESERVOIR_H
#define RESERVOIR_H

using namespace std;


class Reservoir
{
    int num_neurons;
    float tau;
    vector<pair<int, int>> pairs;
    vector<float> weights, x, dot;

public:

    Reservoir(
        int num_neurons,
        float tau,
        const vector<pair<int, int>>& pairs,
        const vector<float>& weights
    );

    void reset();

    vector<float> step(const vector<float>& x_in, float dt);

    vector<float> get_state();
};


#endif
