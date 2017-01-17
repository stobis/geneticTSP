#include "SDL2/SDL.h"
#include "decls.hpp"

SDL_Window* window = NULL;
SDL_Renderer* renderer = NULL;

void initializeSDL()
{
    if (SDL_Init(SDL_INIT_VIDEO) == 0) {
         if (SDL_CreateWindowAndRenderer(640, 480, 0, &window, &renderer) == 0)
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

void drawChromosomeSDL(Chromosome *chromosome)
{
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(renderer);

  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderDrawLine(renderer, 321, 200, 300, 240);
  SDL_RenderDrawLine(renderer, 300, 240, 340, 240);
  SDL_RenderDrawLine(renderer, 340, 240, 320, 200);
  SDL_RenderDrawLine(renderer, 300, 213, 340, 213);
  SDL_RenderDrawLine(renderer, 300, 213, 320, 253);
  SDL_RenderDrawLine(renderer, 340, 213, 320, 253);
  SDL_RenderPresent(renderer);
}
