#ifndef HOST_DECLS
#define HOST_DECLS

#include "decls.hpp"

double dist(Point a, Point b);
double distGraph(int a, int b);

void createGeneration( CUdeviceptr oldGen, CUdeviceptr newGen );
void checkRes(char *message, CUresult res);

extern Point *graph;

extern int graphSize, generationSize, generationLimit;
extern int *oldPaths, *newPaths;
extern Chromosome* oldGeneration;
extern Chromosome* newGeneration;
extern CUfunction breed;
extern CUfunction initializeChromosomes;
extern CUfunction declsFunc;

extern CUdeviceptr devGraph, devOldGeneration, devNewGeneration, devOldPaths, devNewPaths;

#endif
