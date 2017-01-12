#include "cuda.h"
#include <cstdio>
#include <string>
#include <ctime>
#include <algorithm>
#include <cstdlib>

#include "hostDecls.hpp"
#include "hostDefs.cpp"
#include "structDefs.cpp"

void createFirstGeneration(int *paths);

int main(int argv, char *argc[])
{
  srand(time(NULL));
  //Inicjalizajca drivera - za nim uruchomimy jaka kolwiek funkcje z Driver API
  cuInit(0);

  //Pobranie handlera do devica
  //(moze byc kilka urzadzen. Tutaj zakladamy, ze jest conajmniej jedno)
  CUdevice cuDevice;
  CUresult res = cuDeviceGet(&cuDevice, 0);
  if (res != CUDA_SUCCESS)
  {
    printf("cannot acquire device 0\n");
    exit(1);
  }
  //Tworzy kontext
  CUcontext cuContext;
  res = cuCtxCreate(&cuContext, 0, cuDevice);
  if (res != CUDA_SUCCESS)
  {
    printf("cannot create context\n");
    exit(1);
  }

  CUmodule uberModule = (CUmodule) 0;
  res = cuModuleLoad(&uberModule, "cuda.ptx");
  checkRes("cannot load breed module", res);

  res = cuModuleGetFunction(&breed, uberModule, "breed");
  checkRes("cannot acquire kernel handle suma", res);

  res = cuModuleGetFunction(&initializeChromosomes, uberModule, "initializeChromosome");
  checkRes("cannot acquire init module", res);

  res = cuModuleGetFunction(&declsFunc, uberModule, "initializeVariables");
  checkRes("cannot acquire decls module", res);

  res = cuModuleGetFunction(&printCu, uberModule, "printGraph");
  checkRes("cannot acquire print kernel", res);

  scanf("%d %d", &graphSize, &generationLimit);
  generationSize = 2 * graphSize;

  newGeneration = new Chromosome[generationSize];
  oldGeneration = new Chromosome[generationSize];

  graph = new Point[graphSize];
  int a, b;
  for (int i = 0; i < graphSize; ++i)
  {
    scanf("%d%d", &a, &b);
    graph[i].x = a;
    graph[i].y = b;
  }

  res = cuMemAlloc(&devGraph, sizeof(Point) * graphSize);
  checkRes("cannot allocate memory devGraph", res);

  cuMemHostRegister(graph, sizeof(Point) * graphSize, 0);
  res = cuMemcpyHtoD(devGraph, graph, sizeof(Point) * graphSize);
  checkRes("cannot copy the table graph", res);

  res = cuMemAlloc(&devOldGeneration, sizeof(Chromosome) * generationSize);
  checkRes("cannot allocate Asuma", res);

  res = cuMemAlloc(&devNewGeneration, sizeof(Chromosome) * generationSize);
  checkRes("cannot allocate Asuma", res);

  res = cuMemAlloc(&devOldPaths, sizeof(int) * generationSize * graphSize);
  checkRes("cannot allocate OldPaths", res);

  res = cuMemAlloc(&devNewPaths, sizeof(int) * generationSize * graphSize);
  checkRes("cannot allocate NewPaths", res);

  oldPaths = new int[generationSize * graphSize];
  newPaths = new int[generationSize * graphSize];

  createFirstGeneration(oldPaths);
  createFirstGeneration(newPaths);

  res = cuMemcpyHtoD(devOldPaths, oldPaths, sizeof(int) * generationSize * graphSize);
  checkRes("cannot cpy paths", res);
  res = cuMemcpyHtoD(devNewPaths, newPaths, sizeof(int) * generationSize * graphSize);
  checkRes("cannot cpy paths", res);

  void *declsArgs[] = {&devGraph, &devOldGeneration, &devNewGeneration, &devOldPaths, &devNewPaths, &graphSize,
                       &generationSize};
  res = cuLaunchKernel(declsFunc, 1, 1, 1, 1, 1, 1, 0, 0, declsArgs, 0);
  checkRes("cannot run declsFunc kernel", res);
  res = cuCtxSynchronize();
  checkRes("cannot sync after declsFunc", res);


  int threadsPerBlock = 1024;
  int blocksPerGrid = (generationSize + threadsPerBlock - 1) / threadsPerBlock;

  void *initArgs[] = {&devOldGeneration, &devOldPaths};
  res = cuLaunchKernel(initializeChromosomes, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, initArgs, 0);
  checkRes("cannot run init kernel", res);
  res = cuCtxSynchronize();
  checkRes("cannoc sync after init kernel", res);

  cuMemcpyDtoH(newGeneration, devNewGeneration, sizeof(Chromosome));
  int *minPtr = newGeneration[0].path;


  for (int i = 0; i < generationLimit; ++i)
  {
    createGeneration(devOldGeneration, devNewGeneration);
    std::swap(devOldGeneration, devNewGeneration);
  }
  std::swap(devOldGeneration, devNewGeneration);


  res = cuMemcpyDtoH(newGeneration, devNewGeneration, sizeof(Chromosome) * generationSize);
  for (int i = 0; i < generationSize; i++)
  {
    newGeneration[i].path = (int *) ((long long) newGeneration[i].path - (long long) minPtr);
    newGeneration[i].path = (int *) ((long long) newGeneration[i].path + (long long) newPaths);
  }

  for (int i = 0; i < generationSize; i++)
  {
    printf("%d: ", newGeneration[i].pathLength);
    for (int j = 0; j < graphSize; j++)
    {
      printf("%d ", newGeneration[i].path[j]);
    }
    printf("\n");
  }

  cuCtxDestroy(cuContext);
  return 0;
}

void checkRes(char *message, CUresult res)
{
  if (res != CUDA_SUCCESS)
  {
    printf("%s %d\n", message, res);
    exit(1);
  }
}

void createFirstGeneration(int *paths)
{
  for (int i = 0; i < generationSize; ++i)
  {
    for (int j = 0; j < graphSize; ++j)
    {
      paths[i * graphSize + j] = j;
    }
    std::random_shuffle(paths + graphSize * i + 1, paths + graphSize * (i + 1));

  }

}
