double dist(Point a, Point b);
double distGraph(int a, int b);

void createGeneration(CUdeviceptr oldGen, CUdeviceptr newGen);
void checkRes(char *message, CUresult res);

Point *graph;

int graphSize, generationSize, generationLimit;
int *oldPaths, *newPaths;
double mutationRatio;
Chromosome *oldGeneration;
Chromosome *newGeneration;
CUfunction breed;
CUfunction initializeChromosomes;
CUfunction declsFunc;
CUfunction printCu;
CUfunction mutate;

CUdeviceptr devGraph, devOldGeneration, devNewGeneration, devOldPaths, devNewPaths;

CUdeviceptr devCurandStates;
CUdeviceptr devRatiosSums;
