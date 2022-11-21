#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "stdafx.hpp"
using namespace std;
namespace py = pybind11;


PYBIND11_MODULE(_reservoir, obj)
{
    py::class_<Reservoir>(obj, "Reservoir")
        .def(py::init<int, float, vector<pair<int, int>>, vector<float>>())
        .def("reset", &Reservoir::reset)
        .def("step", &Reservoir::step)
        .def("get_state", &Reservoir::get_state);
}
