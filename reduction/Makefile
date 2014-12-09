all: reduction.cu
	nvcc -IchLib -arch sm_20 reduction.cu -o reduction.out
clean:
	rm reduction.out
