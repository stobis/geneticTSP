#include "cuda.h"
#include <cstdio>
#include <string>
#include <ctime>
#include <algorithm>
#include <cstdlib>

#include "decls.hpp"

void createFirstGeneration(){
	for(int i = 0; i < generationSize; ++i){
		oldGeneration[i] = new Chromosome(); 
	}
}


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

    //Tworzy modul z pliku binarnego "radixsort.ptx"
    CUmodule cuModule = (CUmodule)0;
    res = cuModuleLoad(&cuModule, "proj.ptx");
    if (res != CUDA_SUCCESS) {
        printf("cannot load module: %d\n", res);  
        exit(1); 
    }

    //Pobiera handler kernela z modulu
    
    res = cuModuleGetFunction(&breed, cuModule, "breed");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle suma\n");
        exit(1);
    }

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

    CUdeviceptr devGraph, devOldGeneration, devNewgeneration;
    
    res = cuMemAlloc(&devGraph, sizeof(Point)*graphSize);
    if ( res != CUDA_SUCCESS){
        printf("cannot allocate memory devGraph\n");
    }

    cuMemHostRegister(graph, sizeof(Point)*graphSize, 0);
    res = cuMemcpyHtoD(devGraph, graph, sizeof(Point)*graphSize);
    if(res != CUDA_SUCCESS){
        printf("cannot copy the table graph");
    }
   
    res = cuMemAlloc(&devOldGeneration, sizeof(Chromosome)*generationSize); 
    if(res != CUDA_SUCCESS){
        printf("cannot alloc outAdev\n");
    }

    res = cuMemHostRegister(oldGeneration, sizeof(Chromosome)*generationSize, 0);
    if(res != CUDA_SUCCESS){
        printf("cannot register outA\n");
    }

    res = cuMemHostRegister(newGeneration, sizeof(Chromosome)*generationSize, 0);
    if(res != CUDA_SUCCESS){
        printf("cannot register outA\n");
    }

    res = cuMemAlloc(&devOldgeneration, sizeof(Chromosome)*generationSize);
    if(res != CUDA_SUCCESS){
        printf("cannot allocate Asuma\n");
        exit(1);
    }

    res = cuMemAlloc(&devNewgeneration, sizeof(Chromosome)*generationSize);
    if(res != CUDA_SUCCESS){
        printf("cannot allocate Asuma\n");
        exit(1);
    }

    res = cuMemcpyHtoD(devOldgeneration, oldGeneration, sizeof(Chromosome)*generationSize);
    if(res != CUDA_SUCCESS){
        printf("cannot copy the table graph");
    }

    Chromosome theBest;
    for(int i = 0; i < generationLimit; ++i){
    	theBest = createGeneration(devOldgeneration, devNewgeneration)
    }
    cuCtxDestroy(cuContext);
return 0;
}
