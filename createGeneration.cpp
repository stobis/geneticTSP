#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "cuda.h"

#include "hostDecls.hpp"
#include "structDefs.cpp"

void createGeneration( CUdeviceptr oldGen, CUdeviceptr newGen )
{
    printf("BBB\n");
    thrust::device_ptr<Chromosome> p( (Chromosome *) oldGen );
    thrust::device_ptr<Chromosome> q( ((Chromosome *) oldGen) + generationSize );
  thrust::sort( p, q );  // nie wiem czy dziala, znalazlem na necie;

    printf("AAA\n");

  printCudaGraph(oldGen);

  void *breedingArgs[] = {&oldGen, &newGen };

  int threadsPerBlock = 1024;
  int blocksPerGrid = ( generationSize + threadsPerBlock - 1 ) / threadsPerBlock;

  CUresult res = cuLaunchKernel( breed, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, breedingArgs, 0 );
  checkRes("cannot launch breeding", res);
  cuCtxSynchronize();
  checkRes("cannot sync after breed", res);
}

