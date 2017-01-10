struct Chromosome{
	int pathLength;
	int* path;

	Chromosome(){
		pathLength=0;
		path = new int[graphSize];	
		for(int i = 0; i < graphSize; ++i)
			path[i] = i;
		std::random_shuffle(path+1, path+graphSize);
    updatePathLength();
	}

	Chromosome(int pathLength, int* pathx): pathLength(pathLength){
    initializePath(pathx);
	}
  
  Chromosome(int *pathx){
    initializePath(pathx);
    updatePathLength();
  }

	void put(int* pathx){
    pathLength = 0;
		path[0] = pathx[0];
    
		for(int i = 1; i < graphSize; ++i){
			path[i] = pathx[i];
			pathLength += distGraph(path[i-1], path[i]);
		}
    
		pathLength+=distGraph(path[graphSize-1], path[0]);
	}

	~Chromosome(){
		pathLength = 0;
		delete [] path;
	}
  
  void initializePath(int *pathx){
    path = new int[graphSize]; 
    
		for(int i = 0; i < graphSize; ++i){
			path[i] = pathx[i];
		}
  }
  
  void updatePathLength(){
    pathLength = 0;
    
    for(int i = 1; i < graphSize; ++i){
			pathLength += distGraph(path[i-1], path[i]);
		}
		pathLength += distGraph(path[graphSize-1], path[0]);
  }
};

bool operator<( Chromosome a, Chromosome b )
{
  return a.pathLength < b.pathLength;
}

