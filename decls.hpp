#include <curand_kernel.h>

struct Chromosome;

struct Point
{
  double x, y;
};

double dist(Point a, Point b);
double distGraph(int a, int b);

Chromosome *createGeneration( Chromosome *oldGen, Chromosome *newGen );  // returns best of new generation
Chromosome *cross( Chromosome *a, Chromosome *b );                       // returns child of a and b

Point *graph;

curandState *devStates;
struct Chromosome;
int graphSize, generationSize, generationLimit;
Chromosome* oldGeneration[generationSize];
Chromosome* newGeneration[generationSize];
CUfunction breed;

#include "Chromosome.cpp"

double dist(Point a, Point b){
	return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
}

double distGraph(int a, int b){
  return dist(graph[a], graph[b]); 
}

