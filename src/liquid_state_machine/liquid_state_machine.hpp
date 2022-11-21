#ifndef LSM_H
#define LSM_H

using namespace std;


struct LSMParam
{
    int num_input;
    int num_rsrvr;
    int num_syn_input;
    int num_syn_rsrvr;
    float tau_m;
    float v_th;
    float v_rest;
    float v_reset;
    float i_back;
    float i_noise_scale;
    float r;
    vector<bool> is_exc_input;
    vector<bool> is_exc_rsrvr;
    vector<pair<int, int>> pairs_input;
    vector<pair<int, int>> pairs_rsrvr;
    vector<float> t_ref;
    vector<float> tau_decay;
    vector<float> tau_syn_input;
    vector<float> tau_syn_rsrvr;
    vector<float> tau_d;
    vector<float> tau_f;
    vector<float> u0;
    vector<float> a_input;
    vector<float> a_rsrvr;
    vector<float> delay_input;
    vector<float> delay_rsrvr;

    LSMParam(
        int num_input,
        int num_rsrvr,
        int num_syn_input,
        int num_syn_rsrvr,
        float tau_m,
        float v_th,
        float v_rest,
        float v_reset,
        float i_back,
        float i_noise_scale,
        float r,
        const vector<bool>& is_exc_input,
        const vector<bool>& is_exc_rsrvr,
        const vector<pair<int, int>>& pairs_input,
        const vector<pair<int, int>>& pairs_rsrvr,
        const vector<float>& t_ref,
        const vector<float>& tau_decay,
        const vector<float>& tau_syn_input,
        const vector<float>& tau_syn_rsrvr,
        const vector<float>& tau_d,
        const vector<float>& tau_f,
        const vector<float>& u0,
        const vector<float>& a_input,
        const vector<float>& a_rsrvr,
        const vector<float>& delay_input,
        const vector<float>& delay_rsrvr
    );
};


class LiquidStateMachine
{
    LSMParam param;
    int n_step;
    bool log_flg;
    vector<float> u, x, i_input, i_input_delayed, i_rsrvr, i_rsrvr_delayed, i_sum;
    vector<float> s_re, v, trace, t_last, di_input, di_rsrvr, du, dx, dv, dtrace;
    vector<vector<int>> spike_log;
    mt19937 mt;
    uniform_real_distribution<float> rand1;

public:

    LiquidStateMachine(const LSMParam& param, int seed);

    void reset(const vector<float>& init_v);

    vector<float> step(const vector<float>& s_in, float dt);

    void start_log();

    vector<float> get_trace();

    vector<vector<int>> get_spike_log();
};


#endif
