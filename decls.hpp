#include <curand_kernel.h>

struct Chromosome{
	int pathLength;
	int* path;

	Chromosome(){
		pathLength=0;
		path = new int[graphSize];	
		for(int i = 0; i < graphSize; ++i)
			path[i] = i;
		std::random_shuffle(path+1, path+graphSize);
		for(int i = 1; i < graphSize; ++i){
			val = path[i];
			pathLength += odl(graph[path[i-1]], graph[path[i]]);
			tmp = val;
		}
		pathLength += odl(graph[path[graphSize-1]], graph[path[0]]);
	}

	Chromosome(int pathLength, int* pathx): pathLength(pathLength){
		path = new int[graphSize];
		for(int i = 0; i < graphSize; ++i){
			path[i] = pathx[i];
		}
	}

	void put(int* pathx){
		int tmp = *(pathx);
		path[0] = pathx[0];
		for(int i = 1; i < graphSize; ++i){
			path[i] = pathx[x];
			pathLength += odl(path[i-1], path[i]);
			tmp = path[i];
		}
		pathLength+=odl(path[graphSize-1], path[0]);
	}

	~Chromosome(){
		pathLength = 0;
		delete [] path;
	}
};

bool operator<( Chromosome a, Chromosome b )
{
  return a.pathLength < b.pathLength;
}

struct Point
{
  double x, y;
};

double odl(Point a, Point b){
	return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
}

Chromosome *createGeneration( Chromosome *oldGen, Chromosome *newGen );  // returns best of new generation
Chromosome *cross( Chromosome *a, Chromosome *b );                       // returns child of a and b

Point *graph;

curandState *devStates;
struct Chromosome;
int graphSize, generationSize, generationLimit;
Chromosome* oldGeneration;
Chromosome* newGeneration;
CUfunction breed;
