#include<cstdio>
#include<cstdlib>
#include "decls.hpp"

extern "C" {
  __global__
    Chromosome *cross( Chromosome *a, Chromosome *b ) {
      return a;
    }
}

