#include "decls.hpp"

extern __device__  int *oldGeneration, *newGeneration, *oldPaths, *newPaths;
extern __device__  int graphSize, generationSize;
extern __device__ Point *graph;

extern "C" {
__device__ int getRandNorm(int p, int q);
__device__ void cross( Chromosome *a, Chromosome *b, Chromosome *child );

__device__ int calculatePathLength( Chromosome *chromosome, int *path);
__device__ int dist(Point a, Point b);
__device__ int distGraph(int a, int b);
}
