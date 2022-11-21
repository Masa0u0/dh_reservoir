#ifndef SPIKE_ENCODER
#define SPIKE_ENCODER

using namespace std;


class PoissonEncoder
{
    int obs_dim;
    float max_freq;

    mt19937 mt;
    uniform_real_distribution<float> rand_01;

public:

    PoissonEncoder(int obs_dim, float max_freq);

    vector<float> encode(const vector<float>& obs, float dt);
};


class PopSANEncoder
{
    int obs_dim;
    int pop_dim;
    int spike_ts;
    float std;
    float v_reset;
    float v_th;

    int encoder_neuron_dim;
    vector<float> mean;

public:

    PopSANEncoder(
        int obs_dim,
        int pop_dim,
        int spike_ts,
        float std,
        float v_reset,
        float v_th,
        const pair<float, float>& mean_range
    );

    vector<vector<float>> encode(const vector<float>& obs);
};


class LIFEncoder
{
    int obs_dim;
    int pop_dim;
    float std;
    float v_rest;
    float v_reset;
    float v_th;
    float tau_m;
    float amp;

    int encoder_neuron_dim;
    vector<float> mean, pop_volt, pop_spikes;

public:

    LIFEncoder(
        int obs_dim,
        int pop_dim,
        const pair<float, float>& mean_range,
        float std,
        float v_rest,
        float v_reset,
        float v_th,
        float tau_m,
        float amp
    );

    vector<float> encode(const vector<float>& obs, float dt);

    void reset(const vector<float>& init_v);
};


#endif
