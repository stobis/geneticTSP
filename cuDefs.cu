#ifndef CUDECLS_CU
#define CUDECLS_CU


#include "decls.hpp"
#include "structDefs.cpp"
#include "cuDecls.cu"

__device__  int *oldGeneration, *newGeneration, *oldPaths, *newPaths;
__device__  int graphSize, generationSize;
__device__ Point *graph;
    
  
extern "C" {
    __global__
    void initializeVariables( Point *cuGraph, int *cuOldGen, int *cuNewGen, int *cuOldPaths, int *cuNewPaths, int cuGraphSize, int cuGenerationSize) {
        graph = cuGraph;
        oldGeneration = cuOldGen;
        newGeneration = cuNewGen;
        oldPaths = cuOldPaths;
        newPaths = cuNewPaths;
        graphSize = cuGraphSize;
        generationSize = cuGenerationSize;
        printf("%d, %d\n", graphSize, generationSize);
    }
}
   

#endif
