#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "./liquid_state_machine.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_liquid_state_machine, obj)
{
  py::class_<LSMParam>(obj, "LSMParam")
    .def(py::init<
         int, int, int, int, double, double, double, double, double, double, double, vector<bool>,
         vector<bool>, vector<pair<int, int>>, vector<pair<int, int>>, vector<double>,
         vector<double>, vector<double>, vector<double>, vector<double>, vector<double>,
         vector<double>, vector<double>, vector<double>, vector<double>, vector<double>>());

  py::class_<LiquidStateMachine>(obj, "LiquidStateMachine")
    .def(py::init<LSMParam, int>())
    .def("reset", &LiquidStateMachine::reset)
    .def("step", &LiquidStateMachine::step)
    .def("start_log", &LiquidStateMachine::start_log)
    .def("get_trace", &LiquidStateMachine::get_trace)
    .def("get_spike_log", &LiquidStateMachine::get_spike_log);
}
