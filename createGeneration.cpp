#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "cuda.h"

#include "hostDecls.hpp"
#include "structDefs.cpp"

void createGeneration()
{
  CUresult res = cuMemcpyDtoH(oldGeneration, devOldGeneration, sizeof(Chromosome) * generationSize);

  double *ratios = new double[generationSize];
  double *ratiosSums = new double[generationSize];


  for (int i = 0; i < generationSize; i++)
  {
    ratios[i] = 1.0 / (double) oldGeneration[i].pathLength;
  }


  thrust::inclusive_scan(ratios, ratios + generationSize, ratiosSums);


  res = cuMemcpyHtoD(devRatiosSums, ratiosSums, sizeof(double) * generationSize);
  checkRes("cannot copy ratio sums", res);

  void *breedingArgs[] = {&devOldGeneration, &devNewGeneration, &devRatiosSums, &devCurandStates};

  int threadsPerBlock = 1024;
  int blocksPerGrid = (generationSize + threadsPerBlock - 1) / threadsPerBlock;


  res = cuLaunchKernel(breed, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, breedingArgs, 0);
  checkRes("cannot launch breeding", res);
  res = cuCtxSynchronize();
  checkRes("cannot sync after breed", res);

  //printCudaGraph(devNewGeneration);
  //
  res = cuLaunchKernel(mutate, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, breedingArgs, 0);
  checkRes("cannot launch mutating", res);
  res = cuCtxSynchronize();
  checkRes("cannot sync after mutation", res);

  Chromosome *tmp = oldGeneration;
  oldGeneration = newGeneration;
  newGeneration = tmp;

  delete[] ratios;
  delete[] ratiosSums;
}
