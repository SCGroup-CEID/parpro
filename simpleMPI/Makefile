NVFLAGS := -arch=sm_20 -O3
CFLAGS :=
LIBRARIES :=

# Common includes and paths for CUDA
CUDA_PATH=/usr/local/cuda-6.0
CUDA_LIB=$(CUDA_PATH)/lib64
CUDA_INC=$(CUDA_PATH)/include

# Common includes and paths for MPI
# MPI_PATH=

LIBRARIES += -L$(CUDA_LIB) -lcudart -lcudadevrt

MPICC=mpicc
NVCC=nvcc

# Target rules
all: build

build: simpleMPI.out

simpleMPI_cuda.o: simpleMPI.cu
	$(NVCC) $(NVFLAGS) -o $@ -c $<
simpleMPI.o: simpleMPI.c
	$(MPICC) $(CFLAGS) -I$(CUDA_INC) -o $@ -c $<

simpleMPI.out: simpleMPI.o simpleMPI_cuda.o
	$(MPICC) -o $@ $+ $(LIBRARIES)

run: build
	./simpleMPI

clean:
	rm -f simpleMPI simpleMPI.o simpleMPI_cuda.o
