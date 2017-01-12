#include "cuda.h"

#include "hostDecls.hpp"
#include "structDefs.cpp"

void printCudaGraph(CUdeviceptr ptr)
{
    void *args[] = {&ptr};  
    int threadsPerBlock = generationSize;
    int blocksPerGrid = 1;

    CUresult res = cuLaunchKernel( printCu, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args, 0 );
    checkRes("cannot run init kernel", res);
    res = cuCtxSynchronize();
    checkRes("cannoc sync after init kernel", res);
}
