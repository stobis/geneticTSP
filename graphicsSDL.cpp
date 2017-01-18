#include "SDL2/SDL.h"
#include "decls.hpp"
#include "structDefs.cpp"
SDL_Window* window = NULL;
SDL_Renderer* renderer = NULL;

void initializeSDL(int windowLength, int windowHeight)
{
    if (SDL_Init(SDL_INIT_VIDEO) == 0) {
         if (SDL_CreateWindowAndRenderer(windowLength, windowHeight, 0, &window, &renderer) == 0)
         {
            //ALL OK
         }
         else
         {
            printf("Cannot create window\n"); 
         }
    }
    else
    {
        printf("Cannot init SDL\n"); 
    }

}

void destroySDL()
{
  SDL_bool done = SDL_FALSE;

  SDL_Event event;
  while (!done) {
    while (SDL_PollEvent(&event)) {
     if (event.type == SDL_QUIT) {
         done = SDL_TRUE;
     }
    }
  }

  if (renderer) 
  {
    SDL_DestroyRenderer(renderer);
  }
  if (window)
  {
    SDL_DestroyWindow(window);
  }

  SDL_Quit();
}

void drawChromosomeSDL(Chromosome chromosome, Point* drawGraph, int graphSize)
{
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(renderer);

   
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
  int p1, p2;
  p1 = chromosome.path[0];
  for(int i = 1; i < graphSize; ++i){
    p2 = chromosome.path[i];
    SDL_RenderDrawLine(renderer, drawGraph[p1].x, drawGraph[p1].y, drawGraph[p2].x, drawGraph[p2].y);
    p1 = p2;
  }
  SDL_RenderDrawLine(renderer, drawGraph[p1].x, drawGraph[p1].y, drawGraph[0].x, drawGraph[0].y);

  SDL_Rect rect;
  SDL_SetRenderDrawColor(renderer, 0, 255, 0, SDL_ALPHA_OPAQUE);
  for(int i = 0; i < graphSize; ++i){
    rect.x = drawGraph[i].x - 3;
    rect.y = drawGraph[i].y - 3;
    rect.h = rect.w = 6;
    SDL_RenderFillRect(renderer, &rect);
  }


  SDL_RenderPresent(renderer);
}
