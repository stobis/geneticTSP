#ifndef HOST_DECLS
#define HOST_DECLS

#include "decls.hpp"

void createGeneration();
void checkRes(char *message, CUresult res);
void printCudaGraph(CUdeviceptr ptr);

void initializeSDL(int windowLength, int windowHeight);
void destroySDL();
void drawChromosomeSDL(Chromosome chromosome, Point * drawGraph, int graphSize);

extern Point *graph;
extern Point *drawGraph; 

extern int graphSize, generationSize, generationLimit;
extern int *oldPaths, *newPaths;
extern int windowHeight, windowLength;
extern double mutationRatio;
extern Chromosome *oldGeneration;
extern Chromosome *newGeneration;
extern CUfunction breed;
extern CUfunction initializeChromosomes;
extern CUfunction declsFunc;
extern CUfunction printCu;
extern CUfunction mutate;

extern CUdeviceptr devGraph, devOldGeneration, devNewGeneration, devOldPaths, devNewPaths;

extern CUdeviceptr devCurandStates;
extern CUdeviceptr devRatiosSums;

#endif
