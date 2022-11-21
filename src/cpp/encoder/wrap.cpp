#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "./encoder.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_encoder, obj)
{
  py::class_<PoissonEncoder>(obj, "PoissonEncoder")
    .def(py::init<int, double>())
    .def("encode", &PoissonEncoder::encode);

  py::class_<PopSANEncoder>(obj, "PopSANEncoder")
    .def(py::init<int, int, int, double, double, double, pair<double, double>>())
    .def("encode", &PopSANEncoder::encode);

  py::class_<LIFEncoder>(obj, "LIFEncoder")
    .def(py::init<int, int, pair<double, double>, double, double, double, double, double, double>())
    .def("encode", &LIFEncoder::encode)
    .def("reset", &LIFEncoder::reset);
}
