#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "./neuron.hpp"
#include "./synapse.hpp"
#include "./ext_lsm.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_ext_lsm, obj)
{
  py::class_<NeuronParams>(obj, "NeuronParams").def(py::init<>());

  py::class_<SynapseParams>(obj, "SynapseParams").def(py::init<>());

  py::class_<LiquidStateMachine>(obj, "LiquidStateMachine")
    .def(py::init<
         vector<vector<float>>, vector<vector<float>>, NeuronParams, SynapseParams, pair<int, int>,
         float, float, float, bool, float, int>())
    .def("reset", &LiquidStateMachine::reset)
    .def("step", &LiquidStateMachine::step)
    .def("get_num_input_neurons", &LiquidStateMachine::get_num_input_neurons)
    .def("get_num_rsrvr_neurons", &LiquidStateMachine::get_num_rsrvr_neurons)
    .def("input_connectivity", &LiquidStateMachine::input_connectivity)
    .def("rsrvr_connectivity", &LiquidStateMachine::rsrvr_connectivity);
}
