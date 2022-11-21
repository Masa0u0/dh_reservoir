#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "./core.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_reservoir, obj)
{
  py::class_<Edge>(obj, "Edge").def(py::init<int, int, double>());

  py::class_<Reservoir>(obj, "Reservoir")
    .def(py::init<int, double, const vector<Edge>&>())
    .def("reset", &Reservoir::reset)
    .def("step", &Reservoir::step)
    .def("get_state", &Reservoir::getState);
}
