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
            
        // zalozenie: a i b to ciagi zaczynajace sie zerem, wynik zachowa te wlasnosc
        
        int global_thid = blockIdx.x*blockDim.x + threadIdx.x;
        int thid = threadIdx.x;
        
        /* init usages */
        
        bool *visited = new bool[graphSize];
        
        for (int i = 0; i < graphSize; ++i) {;
            visited[i] = false;
        }
        
        int * res = new int[graphSize];
        res[0] = 0; visited[0] = true;
        int resSize = 1;
        
        int P = res[resSize-1]; // niezmiennik: P jest ostatnim wierzcholkiem danego, czesciowego wyniku
        int P1 = 0, P2 = 0;
        
        while (resSize < graphSize) {
            
        
            // P1 i P2 to miejsca gdzie jest P na sciezce w chromosomie a i b
            
            for (int i = 0; i < graphSize; ++i) {
            
                if (a->path[i] == P) {
                    P1 = i;
                }
                if (b->path[i] == P) {
                    P2 = i;
                }
            }
            
            // Szukam pierwszego wolnego wierzcholka po P na listach a i b
            int next1 = P1 + 1, next2 = P2 + 1;
            
            // tam gdzie wskazuja next1 i next2 sa kandydaci do porownania (numery wierzcholkow)
            int candidate1 = -1, candidate2 = -1;
            
            while (next1 < graphSize && visited[a->path[next1]]) {
                next1++;
            }
            
            if (next1 >= graphSize) {   // doszedlem do konca i nie znalazlem nieodwiedzonego w ciagu
                // ergo musze szukac w zbiorze {1....n-1}
                for (int i = 0; i < graphSize; ++i) {
                    if (!visited[i]) {
                        candidate1 = i;
                        break;
                    }
                }
            }
            
            while (next2 < graphSize && visited[b->path[next2]]) {
                next2++;
            }
            
            if (next2 >= graphSize) {   // doszedlem do konca i nie znalazlem nieodwiedzonego w ciagu
                // ergo musze szukac w zbiorze {1....n-1}
                for (int i = 0; i < graphSize; ++i) {
                    if (!visited[i]) {
                        candidate2 = i;
                        break;
                    }
                }
            }
            
            // jesli na liscie a lub b byl wolny wierzcholek. to nie szukalem dodatkowo, wiec moge przepisac wartosc
            if (candidate1 < 0) candidate1 = a->path[next1];
            if (candidate2 < 0) candidate2 = b->path[next2];
            
            // teraz musze miec odleglosci miedzy P a cand1 i P a cand2
            
            int dist1 = distGraph(P, candidate1);
            int dist2 = distGraph(P, candidate2);
            
            // Dodaje lepsza opcje do wyniku
            
            if (dist1 < dist2) {
            
                res[resSize++] = candidate1;
                visited[candidate1] = true;
            }
            else {
            
                res[resSize++] = candidate2;
                visited[candidate2] = true;
            }
            
            
            // Aktualizuje niezmiennik
            P = res[resSize-1];
            
        }
        
        // Przepisuje wynik tam gdzie trzeba
        for (int i = 0; i < resSize; ++i) {
            child->path[i] = res[i];
        }
        child->pathLength = calculatePathLength(child, child->path);

        delete[] res;
        delete[] visited;
        
        return;
     
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
    }

  __global__
    void initializeChromosome( Chromosome *chromosomes, int *paths) {
      int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if(thid >= generationSize) return;

      chromosomes[thid].path = paths+(graphSize*thid);
      chromosomes[thid].pathLength = calculatePathLength(chromosomes+thid, paths+thid*graphSize);

      // Simple cross test for debug only!
      __syncthreads();
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
        //return 1;
    	return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
    }
    
  __device__
    int distGraph(int a, int b){
      return dist(graph[a], graph[b]); 
    }

 __global__
    void printGraph(Chromosome* g){
      int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if(thid > generationSize ) return;

      printf("%d: len=%d, path=\n", thid, g[thid].pathLength);
      for(int i=0; i<graphSize; i++)
      {
        printf("%d, %d: %d \n", thid, i, g[thid].path[i]);   
      }
      
    }

}
