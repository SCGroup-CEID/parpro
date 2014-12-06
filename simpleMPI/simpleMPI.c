#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// MPI include
#include <mpi.h>

// Shut down MPI cleanly if something goes wrong
void my_abort(int err)
{
    printf("Test FAILED\n");
    MPI_Abort(MPI_COMM_WORLD, err);
}

// Error handling macros
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { \
        my_abort(-1); }


extern void initData(float *data, int dataSize);
extern void computeGPU(float *hostData, int blockSize, int gridSize);

// Host code
// No CUDA here, only MPI
int main(int argc, char *argv[])
{
    // Dimensions of the dataset
    int blockSize = 256;
    int gridSize = 10000;
    int dataSizePerNode = gridSize * blockSize;

    // Initialize MPI state
    MPI_CHECK(MPI_Init(&argc, &argv));

    // Get our MPI node number and node count
    int commSize, commRank;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));

    // Generate some random numbers on the root node (node 0)
    int dataSizeTotal = dataSizePerNode * commSize;
    float *dataRoot = NULL;
    float *dev = NULL;
    cudaMalloc((void **)&dev, 1024 * sizeof(float));
    if (commRank == 0)  // Are we the root node?
    {
	printf("Running on %d nodes\n",commSize);
	dataRoot = (float*) malloc(dataSizeTotal*sizeof(float));
        initData(dataRoot, dataSizeTotal);
    }

    // Allocate a buffer on each node
    float *dataNode = (float*) malloc(dataSizePerNode*sizeof(float));

    // Dispatch a portion of the input data to each node
    MPI_CHECK(MPI_Scatter(dataRoot,
                          dataSizePerNode,
                          MPI_FLOAT,
                          dataNode,
                          dataSizePerNode,
                          MPI_FLOAT,
                          0,
                          MPI_COMM_WORLD));

    if (commRank == 0)
    {
        // No need for root data any more
        free(dataRoot);
    }

    // On each node, run computation on GPU
    computeGPU(dataNode, blockSize, gridSize);

    // Reduction to the root node, computing the sum of output elements
    float sumNode = sum(dataNode, dataSizePerNode);
    float sumRoot;

    MPI_CHECK(MPI_Reduce(&sumNode, &sumRoot, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));

    if (commRank == 0)
    {
        float average = sumRoot / dataSizeTotal;
	printf("Average of square root is %f\n",average);
    }

    // Cleanup
    free(dataNode);
    MPI_CHECK(MPI_Finalize());

    if (commRank == 0)
    {
        printf("PASSED\n");
    }

    return 0;
}

