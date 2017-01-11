#ifndef CUDECLS_CU
#define CUDECLS_CU

#include "decls.hpp"

__device__ int *oldGeneration, *newGeneration, *oldPaths, *newPaths;
__device__ int graphSize, generationSize;
__device__ Point *graph;

extern "C" {

  __global__
    void initializeVariables( int *cuGraph, int *cuOldGen, int *cuNewGen, int *cuOldPaths, int *cuNewPaths, int graphSize, int generationSize) {
        graph = cuGraph;
        oldGeneration = cuOldGen;
        newGeneration = cuNewGen;
        oldPaths = cuOldPaths;
        newPaths = cuNewPaths;
    }
}
   

#endif
