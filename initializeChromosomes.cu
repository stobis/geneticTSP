#include<cstdio>
#include<cstdlib>

#include "cuDecls.cu"

extern "C" {

  __global__
    void initializeChromosome( Chromosome *chromosomes, int *paths) {
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
      return pathLength;
    }

  __device__
    int dist(Point a, Point b){
    	return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
    }
    
  __device__
    int distGraph(int a, int b){
      return dist(graph[a], graph[b]); 
    }
}

