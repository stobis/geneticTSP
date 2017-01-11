#include <thrust/sort.h>
#include "cuda.h"

#include "hostDecls.hpp"

void createGeneration( CUdeviceptr oldGen, CUdeviceptr newGen )
{
  //thrust::sort( (Chromosome *) oldGen, ( ( Chromosome * ) oldGen ) + generationSize );  // nie wiem czy dziala, znalazlem na necie;

  void *breedingArgs[] = {&oldGen, &newGen };

  int threadsPerBlock = 1024;
  int blocksPerGrid = ( generationSize + threadsPerBlock - 1 ) / threadsPerBlock;

  CUresult res = cuLaunchKernel( breed, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, breedingArgs, 0 );
  checkRes("cannot launch breeding", res);
}
