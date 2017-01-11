#include<cstdio>
#include<cstdlib>
#include <curand_kernel.h>

#include "decls.hpp"
#include "cuDecls.cu"
#include "structDefs.cpp"

extern "C" {
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
}

