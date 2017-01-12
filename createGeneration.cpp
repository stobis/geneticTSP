#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "cuda.h"

#include "hostDecls.hpp"
#include "structDefs.cpp"

void createGeneration(CUdeviceptr oldGen, CUdeviceptr newGen)
{
  printf("BBB\n");
  thrust::device_ptr <Chromosome> p = thrust::device_pointer_cast((Chromosome *) oldGen);
  //thrust::sort( p, p+generationSize );  // nie wiem czy dziala, znalazlem na necie;

  printf("AAA\n");

  printCudaGraph(oldGen);

  void *breedingArgs[] = {&oldGen, &newGen};

  int threadsPerBlock = 1024;
  int blocksPerGrid = (generationSize + threadsPerBlock - 1) / threadsPerBlock;

  CUresult res = cuLaunchKernel(breed, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, breedingArgs, 0);
  checkRes("cannot launch breeding", res);
  cuCtxSynchronize();
  checkRes("cannot sync after breed", res);
}

