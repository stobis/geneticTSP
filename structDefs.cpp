#include "cuda.h"

struct Point
{
  int x, y;
};

struct Chromosome
{
  double pathLength;
  int *path;
};

