#ifndef CUDECLS_CU
#define CUDECLS_CU


#include <curand_kernel.h>
#include "decls.hpp"
#include "structDefs.cpp"

extern "C" {


    __device__ int *oldGeneration, *newGeneration, *oldPaths, *newPaths;
    __device__ int graphSize, generationSize;
    __device__ Point *graph;
    
    __device__ void cross( Chromosome *a, Chromosome *b, Chromosome *child );
    __device__ int getRandNorm(int p, int q);

    __device__ int calculatePathLength( Chromosome *chromosome, int *path);
    __device__ int dist(Point a, Point b);
    __device__ int distGraph(int a, int b);
    
    __global__
    void initializeVariables( Point *cuGraph, int *cuOldGen, int *cuNewGen, int *cuOldPaths, int *cuNewPaths, int graphSize, int generationSize) {
        graph = cuGraph;
        oldGeneration = cuOldGen;
        newGeneration = cuNewGen;
        oldPaths = cuOldPaths;
        newPaths = cuNewPaths;
    }
}
   

#endif
