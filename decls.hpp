#include <curand_kernel.h>

struct Chromosome
{
  int *path;
  int pathLength;
};
bool operator<( Chromosome a, Chromosome b )
{
  return a.pathLength < b.pathLength;
}

struct Point
{
  double x, y;
};

Chromosome *createGeneration( Chromosome *oldGen, Chromosome *newGen );  // returns best of new generation
Chromosome *cross( Chromosome *a, Chromosome *b );                       // returns child of a and b

Point *graph;
int V;
int generationSize;

curandState *devStates;
