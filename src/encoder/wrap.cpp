#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "./encoder.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_encoder, obj)
{
  py::class_<PoissonEncoder>(obj, "PoissonEncoder")
    .def(py::init<int, float>())
    .def("encode", &PoissonEncoder::encode);

  py::class_<PopSANEncoder>(obj, "PopSANEncoder")
    .def(py::init<int, int, int, float, float, float, pair<float, float>>())
    .def("encode", &PopSANEncoder::encode);

  py::class_<LIFEncoder>(obj, "LIFEncoder")
    .def(py::init<int, int, pair<float, float>, float, float, float, float, float, float>())
    .def("encode", &LIFEncoder::encode)
    .def("reset", &LIFEncoder::reset);
}
