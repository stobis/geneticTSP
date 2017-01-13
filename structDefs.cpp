#include "cuda.h"

struct Point
{
  double x, y;
};

struct Chromosome
{
  double pathLength;
  int *path;
};

