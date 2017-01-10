#include<cstdio>
#include<cstdlib>
#include "decls.hpp"

extern "C" {
  __global__
    void breed(int* oldGen, int* newGen, int generationSize, int V, curandState *state){
      int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if(thid > generationSize ) return;
      curand_init(0, thid, 0, 0);
      
      int test = rand();

      int parentA = getRandNorm(0, V, &state[thid]);
      int parentB = getRandNorm(0, V, &state[thid]);
      cross(oldGen+parentA, oldGen+parentB, newGen+thid, V);
    }

  __global__
    int getRandNorm(int p, int q, curandState &state)
    {
      double x = curand_normal(statae, 0, (q-p)/4);
      if( x < 0 ) x = -x;

      int res = x;
      if(res >=q )
        res = 0;

      return res+p;
    }
}

