#include <cmath>

#include "./util.hpp"

double rbf_kernel(double x, double y, double std)
{
  return exp(-pow(x - y, 2.) / (2. * pow(std, 2.)));
}
