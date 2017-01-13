#include "cuda.h"

struct Point
{
  int x, y;
};

struct Chromosome
{
  int pathLength;
  int *path;
};

