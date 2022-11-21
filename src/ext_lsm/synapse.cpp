#include "stdafx.hpp"

using namespace std;


// SynapseParams

SynapseParams::SynapseParams()
{
    u0_mean = 0.5; u0_sd = 0.25;
    tau_d_mean = 0.11; tau_d_sd = 0.055;
    tau_f_mean = 0.005; tau_f_sd = 0.0025;
    c = 0.05 * exp(5.);
}


DynamicSynapse::DynamicSynapse(const SynapseParams& params, int seed)
{
    c = params.c;

    mt19937 mt(seed);
    normal_distribution<float> randn_u0(params.u0_mean, params.u0_sd);
    normal_distribution<float> randn_tau_d(params.tau_d_mean, params.tau_d_sd);
    normal_distribution<float> randn_tau_f(params.tau_f_mean, params.tau_f_sd);

    while (true)
    {
        float cnd = randn_u0(mt);
        if (abs(cnd - params.u0_mean) < params.u0_sd)
        {
            u0 = cnd;
            break;
        }
    }
    while (true)
    {
        float cnd = randn_tau_d(mt);
        if (abs(cnd - params.tau_d_mean) < params.tau_d_sd)
        {
            tau_d = cnd;
            break;
        }
    }
    while (true)
    {
        float cnd = randn_tau_f(mt);
        if (abs(cnd - params.tau_f_mean) < params.tau_f_sd)
        {
            tau_f = cnd;
            break;
        }
    }
}

void DynamicSynapse::reset()
{
    u = u0;
    x = 1.;
    w = 0.;
    amp = 0.;
    du = 0.;
    dx = 0.;
    dw = 0.;
}

void DynamicSynapse::calc_grad(float dt)
{
    float spike = pre_neuron->get_spike();
    du = (-dt / tau_f) * u + u0 * (1. - u) * spike;
    dx = (dt / tau_d) * (1. - x) - u * x * spike;
    dw = (pre_neuron->get_trace() * c * exp(-w) - 1.) * dt;
}

void DynamicSynapse::update()
{
    u += du;
    x += dx;
    w += dw;
    amp = w * u * x;

    // cout << "u: " << u << ", x: " << x << ", w: " << w << ", amp: " << amp << endl;
}


float DynamicSynapse::get_amp()
{
    return amp;
}
