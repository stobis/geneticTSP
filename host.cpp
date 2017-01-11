#include "cuda.h"
#include <cstdio>
#include <string>
#include <ctime>
#include <algorithm>
#include <cstdlib>

#include "decls.hpp"

void createFirstGeneration(int *paths);
void checkRes(char *message, CUresult res);

int main(int argv, char* argc[]){
	srand (time(NULL));
    //Inicjalizajca drivera - za nim uruchomimy jaka kolwiek funkcje z Driver API
    cuInit(0);
    
    //Pobranie handlera do devica
    //(moze byc kilka urzadzen. Tutaj zakladamy, ze jest conajmniej jedno)
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if(res != CUDA_SUCCESS){
        printf("cannot acquire device 0\n"); 
        exit(1);
    }
    //Tworzy kontext
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        printf("cannot create context\n");
        exit(1);
    }

    CUmodule breedModule = (CUmodule)0;
    res = cuModuleLoad(&breedModule, "breed.ptx");
    checkRes("cannot load breed module", res);  

    res = cuModuleGetFunction(&breed, breedModule, "breed");
    checkRes("cannot acquire kernel handle suma", res);


    CUmodule initModule = (CUmodule)0;
    res = cuModuleLoad(&initModule, "initializeChromosomes.ptx");
    checkRes("cannot load init module", res);    

    res = cuModuleGetFunction(&initializeChromosomes, initModule, "initializeChromosomes");
    checkRes("cannot acquire init module", res);

    CUmodule declsModule = (CUmodule)0;
    res = cuModuleLoad(&declsModule, "cuDecls.ptx");
    checkRes("cannot load decls module", res);

    res = cuModuleGetFunction(&declsFunc, declsModule, "initializeVariables");
    checkRes("cannot acquire decls module", res);

    scanf("%d", &graphSize, &generationLimit);
    generationSize = 2*graphSize;
    newGeneration = new Chromosome[generationSize];
    oldGeneration = new Chromosome[generationSize];
    createFirstGeneration(); // nie wiem czy potrzebne
    graph = new Point[graphSize];
    double a, b;
    for(int i = 0; i < graphSize; ++i){
    	scanf("%d%d", &a, &b);
    	graph[i] = new Point(a, b);
    }

    res = cuMemAlloc(&devGraph, sizeof(Point)*graphSize);
    checkRes("cannot allocate memory devGraph", res);

    cuMemHostRegister(graph, sizeof(Point)*graphSize, 0);
    res = cuMemcpyHtoD(devGraph, graph, sizeof(Point)*graphSize);
    checkRes("cannot copy the table graph", res);
   
    res = cuMemAlloc(&devOldgeneration, sizeof(Chromosome)*generationSize);
    checkRes("cannot allocate Asuma", res);

    res = cuMemAlloc(&devNewgeneration, sizeof(Chromosome)*generationSize);
    checkRes("cannot allocate Asuma", res);

    res = cuMemAlloc(&devOldPaths, sizeof(int)*generationSize*graphSize);
    checkRes("cannot allocate OldPaths", res);

    res = cuMemAlloc(&devNewPaths, sizeof(int)*generationSize*graphSize);
    checkRes("cannot allocate NewPaths", res);

    oldPaths = new int[generationSize*graphSize];
    newPaths = new int[generationSize*graphSize];

    createGeneration(oldPaths);
    createGeneration(newPaths);
    
    res = cuLaunchKernel( declsFunc, 1, 1, 1, 1, 1, 1, 0, 0, 
            (void*){ &devGraph, &devOldgeneration, &devNewgeneration, &devOldPaths, &devNewPaths, &graphSize, &generationSize }, 0 );


    
    int threadsPerBlock = 1024;
    int blocksPerGrid = ( generationSize + threadsPerBlock - 1 ) / threadsPerBlock;

    void *initArgs[] = { &devOldgeneration, &devOldPaths };
    res = cuLaunchKernel( initializeChromosomes, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, initArgs, 0 );
    

    Chromosome theBest;
    for(int i = 0; i < generationLimit; ++i){
    	theBest = createGeneration(devOldgeneration, devNewgeneration)
    }
    cuCtxDestroy(cuContext);
return 0;
}

void checkRes(char *message, CUresult res)
{
    if(res != CUDA_SUCCESS)
    {
        printf("%s\n", message);   
        exit(1);
    }
}

void createFirstGeneration(int *paths){
    for(int i = 0; i < generationSize; ++i)
    {
        for(int j = 0; j < graphSize; ++j)
        {
    			paths[ i*graphSize + j ] = j;
        }
    		std::random_shuffle(paths+graphSize*i, paths+graphSize*(i+1));
    }

}
