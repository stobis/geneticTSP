#include "cuda.h"
#include <curand_kernel.h>
#include <cstdio>
#include <string>
#include <ctime>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <unistd.h>

#include "hostDecls.hpp"
#include "hostDefs.cpp"
#include "structDefs.cpp"

void createFirstGeneration(int *paths);
void createDrawGraph(double minX, double minY, double maxX, double maxY, int windowHeight, int windowLength);

bool operator<(const Chromosome a, const Chromosome b);

int main(int argv, char *argc[])
{
  int windowLength = 640;
  int windowHeight = 480;
  initializeSDL(windowLength, windowHeight);
  
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

  res = cuModuleGetFunction(&mutate, uberModule, "mutate");
  checkRes("cannot acquire mutate kernel", res);

  scanf("%d %d %lf", &graphSize, &generationLimit, &mutationRatio);
  generationSize = 2 * graphSize;

  newGeneration = new Chromosome[generationSize];
  oldGeneration = new Chromosome[generationSize];

  graph = new Point[graphSize];
  drawGraph = new Point[graphSize];
  double a, b;
  double minX, minY, maxX, maxY;
  minX = minY = INT_MAX;
  maxX = maxY = INT_MIN;
  for (int i = 0; i < graphSize; ++i)
  {
    scanf("%lf %lf", &a, &b);
    graph[i].x = drawGraph[i].x = a;
    graph[i].y = drawGraph[i].y = b;
    minX = std::min(minX, a);
    minY = std::min(minY, b);
    maxX = std::max(maxX, a);
    maxY = std::max(maxY, b);
  }

  createDrawGraph(minX, minY, maxX, maxY, windowHeight, windowLength);

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

  res = cuMemAlloc(&devRatiosSums, sizeof(double) * generationSize);
  checkRes("cannot alloc ratio sums", res);

  res = cuMemAlloc(&devCurandStates, sizeof(curandState) * generationSize);
  checkRes("cannot allocate cuRandStates", res);

  oldPaths = new int[generationSize * graphSize];
  newPaths = new int[generationSize * graphSize];

  createFirstGeneration(oldPaths);
  createFirstGeneration(newPaths);

  res = cuMemcpyHtoD(devOldPaths, oldPaths, sizeof(int) * generationSize * graphSize);
  checkRes("cannot cpy paths", res);
  res = cuMemcpyHtoD(devNewPaths, newPaths, sizeof(int) * generationSize * graphSize);
  checkRes("cannot cpy paths", res);

  void *declsArgs[] = {&devGraph, &devOldGeneration, &devNewGeneration, &devOldPaths, &devNewPaths, &graphSize,
                       &generationSize, &mutationRatio};
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


  void *initArgsNew[] = {&devNewGeneration, &devNewPaths};
  res = cuLaunchKernel(initializeChromosomes, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, initArgsNew, 0);
  checkRes("cannot run init kernel", res);
  res = cuCtxSynchronize();
  checkRes("cannoc sync after init kernel", res);


  res = cuMemcpyDtoH(newGeneration, devNewGeneration, sizeof(Chromosome));
  checkRes("cannot copt new Generation 2", res);
  res = cuMemcpyDtoH(oldGeneration, devOldGeneration, sizeof(Chromosome));
  checkRes("cannot copy old Generation 2", res);

  int *minPtr[2];
  //if(generationLimit % 2 ) //dont change that ever
  minPtr[0] = newGeneration[0].path;
  minPtr[1] = oldGeneration[0].path;

  CUdeviceptr pathsToGet[2];
 // if( generationLimit % 2 == 0 ) //dont change that ever
  pathsToGet[0] = devNewPaths;
 // else
  pathsToGet[1] = devOldPaths;
  int whichGeneration = 100;
  int * theBestPath = new int[graphSize];
  for (int i = 0; i < generationLimit; ++i)
  {
    createGeneration();
    if(i % whichGeneration == 0){
      res = cuMemcpyDtoH(newGeneration, devNewGeneration, sizeof(Chromosome) * generationSize);
      checkRes("cannot copy New Generation Back", res);
      res = cuMemcpyDtoH(newPaths, pathsToGet[i%2], sizeof(int)*generationSize*graphSize);
      checkRes("cannot copy paths back", res);
      for (int j = 0; j < generationSize; j++)
      {
        newGeneration[j].path = (int *) ((long long) newGeneration[j].path - (long long) minPtr[i%2]);
        newGeneration[j].path = (int *) ((long long) newGeneration[j].path + (long long) newPaths);
      }

      std::sort(newGeneration, newGeneration+generationSize);
      
      drawChromosomeSDL(newGeneration[0], drawGraph, graphSize);
    }
    std::swap(devOldGeneration, devNewGeneration);
    std::swap(oldGeneration, newGeneration);
  }
 /* std::swap(devOldGeneration, devNewGeneration);

  CUdeviceptr pathsToGet;
  if( generationLimit % 2 ) //dont change that ever
    pathsToGet = devNewPaths;
  else
    pathsToGet = devOldPaths;
    

  res = cuMemcpyDtoH(newGeneration, devNewGeneration, sizeof(Chromosome) * generationSize);
  checkRes("cannot copy New Generation Back", res);

  res = cuMemcpyDtoH(newPaths, pathsToGet, sizeof(int)*generationSize*graphSize);
  checkRes("cannot copy paths back", res);
  
  for (int i = 0; i < generationSize; i++)
  {
    newGeneration[i].path = (int *) ((long long) newGeneration[i].path - (long long) minPtr);
    newGeneration[i].path = (int *) ((long long) newGeneration[i].path + (long long) newPaths);
  }

  std::sort(newGeneration, newGeneration+generationSize);
  std::reverse(newGeneration, newGeneration+generationSize);

  drawChromosomeSDL(newGeneration[0], drawGraph, graphSize);
  
  for (int i = 0; i < generationSize; i++)
  {
    printf("%lf: ", newGeneration[i].pathLength);
    for (int j = 0; j < graphSize; j++)
    {
      printf("%d ", newGeneration[i].path[j]);
    }
    printf("\n");
  }*/

  cuCtxDestroy(cuContext);

  printf("All is done\n");

  destroySDL();
  return 0;
}

void checkRes(char *message, CUresult res)
{
  if (res != CUDA_SUCCESS)
  {
    const char *ptr;
    cuGetErrorString(res, &ptr);
    printf("%s: \"%s\"\n", message, ptr);
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

bool operator<(const Chromosome a, const Chromosome b)
{
  return a.pathLength < b.pathLength;
}

void createDrawGraph(double minX, double minY, double maxX, double maxY, int windowHeight, int windowLength)
{
  maxX = maxX - minX;
  for(int i = 0; i < graphSize; ++i){
    drawGraph[i].x = (drawGraph[i].x - minX);
    if(maxX > 0) 
      drawGraph[i].x *= ((static_cast<double>(windowLength - 100)/(maxX)));
  }
  maxY = maxY - minY;
  for(int i = 0; i < graphSize; ++i){
    drawGraph[i].y =  (drawGraph[i].y - minY);
    if(maxY > 0) 
      drawGraph[i].y *= ((static_cast<double>(windowHeight - 100)/(maxY)));
  }
  for(int i = 0; i < graphSize; ++i){
    drawGraph[i].x += 50;
    drawGraph[i].y += 50;
  }
}
