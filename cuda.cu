#include "structDefs.cpp"
#include <curand_kernel.h>


extern "C" {

__device__ int getRandNorm(int p, int q);
__device__ void cross( Chromosome *a, Chromosome *b, Chromosome *child );

__device__ int calculatePathLength( Chromosome *chromosome, int *path);
__device__ int dist(Point a, Point b);
__device__ int distGraph(int a, int b);

__device__  int *oldGeneration, *newGeneration, *oldPaths, *newPaths;
__device__  int graphSize, generationSize;
__device__ Point *graph;

   __global__
    void breed(Chromosome* oldGen, Chromosome* newGen){
      int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if(thid > generationSize ) return;
      curand_init(1234, thid, 0, (curandStateTest_t *)0);
      
      int parentA = getRandNorm(0, graphSize);
      int parentB = getRandNorm(0, graphSize);

      cross(oldGen+parentA, oldGen+parentB, newGen+thid);
    }

  __device__
    int getRandNorm(int p, int q)
    {
      double x = curand_normal((curandState_t *)0) * (q-p)/4;
      if( x < 0 ) x = -x;

      int res = x;
      if(res >=q )
        res = 0;

      return res+p;
    }
   
    __device__
    void cross( Chromosome *a, Chromosome *b, Chromosome *child )
    {
        //TODO
    }

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

  __global__
    void initializeChromosome( Chromosome *chromosomes, int *paths) {
        printf("%d\n", generationSize);
      int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if(thid >= generationSize) return;

      chromosomes[thid].path = paths+(graphSize*thid);
      chromosomes[thid].pathLength = calculatePathLength(chromosomes+thid, paths+thid*graphSize);

      printf("DUPA: %d %d\n", thid, chromosomes[thid].pathLength);
    }

  __device__
    int calculatePathLength( Chromosome *chromosome, int *path) {
      int pathLength = 0;
        for(int i = 1; i < graphSize; ++i){
    			pathLength += distGraph(path[i-1], path[i]);
    		}
    		pathLength += distGraph(path[graphSize-1], path[0]);
        printf("PUPA\n");
      return pathLength;
    }

  __device__
    int dist(Point a, Point b){
        return 1;
    	return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
    }
    
  __device__
    int distGraph(int a, int b){
      return dist(graph[a], graph[b]); 
    }
}
