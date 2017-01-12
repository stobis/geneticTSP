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

bool operator<(const Chromosome a, const Chromosome b)
{
  return a.pathLength < b.pathLength;
}
