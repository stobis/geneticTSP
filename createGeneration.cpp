#include <thrust/sort.h>
#include "cuda.h"
#include "decls.hpp"

Chromosome *createGeneration( CUdeviceptr oldGen, CUdeviceptr newGen )
{
  thrust::sort( (Chromosome *) oldGen, ( ( Chromosome * oldGen ) ) + generationSize );  // nie wiem czy dziala, znalazlem na necie;

  void *breedingArgs[] = {&oldGen, &newGen, &generationSize, &graphSize, &devStates};

  int threadsPerBlock = 1024;
  int blocksPerGrid = ( generationSize + threadsPerBlock - 1 ) / threadsPerBlock;

  res = cuLaunchKernel( breed, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, breedingArgs, 0 );
}
