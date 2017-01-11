#ifndef HOST_DECLS
#define HOST_DECLS

#include <curand_kernel.h>
#include "decls.hpp"

double dist(Point a, Point b);
double distGraph(int a, int b);

Chromosome *createGeneration( Chromosome *oldGen, Chromosome *newGen );  // returns best of new generation
Chromosome *cross( Chromosome *a, Chromosome *b );                       // returns child of a and b

Point *graph;

curandState *devStates;
int graphSize, generationSize, generationLimit;
Chromosome* oldGeneration[generationSize];
Chromosome* newGeneration[generationSize];
CUfunction breed;
CUfunction initializeChromosomes;
CUfunction declsFunc;

CUdeviceptr devGraph, devOldGeneration, devNewgeneration, devOldPaths, devNewPaths;

#endif
