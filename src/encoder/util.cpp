#include <cmath>

#include "./util.hpp"

float rbf_kernel(float x, float y, float std)
{
  return exp(-pow(x - y, 2.) / (2. * pow(std, 2.)));
}
