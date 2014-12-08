all: reduction.cu
	nvcc -I../chLib -arch sm_20 reduction.cu -o reduction.out
clean:
	rm reduction.out
