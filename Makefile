CUDA_INSTALL_PATH ?= /usr/local/cuda
VER = 
CXX := /usr/bin/g++$(VER)
CC := /usr/bin/gcc$(VER)
LINK := /usr/bin/g++$(VER) -fPIC
CCPATH := ./gcc$(VER)
NVCC  := $(CUDA_INSTALL_PATH)/bin/nvcc -ccbin $(CCPATH)

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Libraries
LIB_CUDA := -L/usr/lib/nvidia-current -lcuda


# Options
NVCCOPTIONS = -arch sm_20 -ptx

# Common flags
COMMONFLAGS += $(INCLUDES) -w
NVCCFLAGS += $(COMMONFLAGS) $(NVCCOPTIONS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)



CUDA_OBJS = cuda.ptx
OBJS = host.cpp.o createGeneration.cpp.o printCudaGraph.cpp.o graphicsSDL.cpp.o
TARGET = solution.x
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA) -L sdlLib/lib/ -l SDL2 -Wl,-rpath=sdlLib/lib


.SUFFIXES:	.c	.cpp	.cu	.o	
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I sdlLib/include/ -L sdlLib/lib/ -l SDL2 -Wl,-rpath=sdlLib/lib

$(TARGET): prepare $(OBJS) $(CUDA_OBJS)
	$(LINKLINE)

clean:
	rm -rf $(TARGET) *.o *.ptx

prepare:
	rm -rf $(CCPATH);\
	mkdir -p $(CCPATH);\
	ln -s $(CXX) $(CCPATH)/g++;\
	ln -s $(CC) $(CCPATH)/gcc;
