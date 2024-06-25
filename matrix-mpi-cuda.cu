#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C"
{
extern void setCudaDevice(int myRank);
extern void matrixMultiplyChunk(int myRank, float* chunkData, float* b, float* chunkResult,
                            size_t numCellsInChunk, int threadsCount,
                            int m, int n, int k);
}

// Gets the cuda device count and sets the cuda device
// so that cuda is properly configured for the process
void setCudaDevice(int myRank) {
   int cE;
   int cudaDeviceCount;
   if ((cE = cudaGetDeviceCount(&cudaDeviceCount)) != cudaSuccess) {
      printf(" Unable to determine cuda device count, error is %d, count is %d\n",
            cE, cudaDeviceCount);
      exit(-1);
   }
   if ((cE = cudaSetDevice(myRank % cudaDeviceCount)) != cudaSuccess) {
      printf(" Unable to have rank %d set to cuda device %d, error is %d \n",
            myRank, (myRank % cudaDeviceCount), cE);
      exit(-1);
   }
}

// Uses CUDA threads to multiply matrices chunkData and b
// and stores results of multiplication chunkResult
// myRank - for debugging information
// m - number of rows of chunkData and chunkResult
// n - number of columns of b and chunkResult
// k - number of columns of chunkData and number of rows of b
__global__ void matrixMultiplyChunk_kernel(int myRank, float* chunkData, float* b, float* chunkResult,
                                        int m, int n, int k) {
    int device;
    cudaGetDevice(&device);

    // hardcoded parallelized CUDA matrix multiply with CUDA threads
    size_t indexInChunk, z;
    size_t numCellsInChunk = m * k;
    for (indexInChunk = (blockIdx.x * blockDim.x) + threadIdx.x; indexInChunk < numCellsInChunk; indexInChunk += blockDim.x * gridDim.x) {
        size_t rowInChunk = indexInChunk / k;
        size_t rowOffsetChunk = rowInChunk * k;   // the offset to column 0 of current row in chunk
        size_t columnInChunk = indexInChunk - rowOffsetChunk;
        for (z = 0; z < k; z++) { // column offset of chunkData, row offset of b
            size_t rowOffsetb = z * k;   // the offset to column 0 of row z in b
            chunkResult[indexInChunk] += chunkData[rowOffsetChunk + z] * b[rowOffsetb + columnInChunk];
        }
    }
}

// Uses kernel function to execute matrix multiple of matrices chunkData and b
// and stores results in chunkResult
void matrixMultiplyChunk(int myRank, float* chunkData, float* b, float* chunkResult,
                        size_t numCellsInChunk, int threadsCount,
                        int m, int n, int k) {
    size_t numBlocks = (numCellsInChunk + threadsCount - 1) / threadsCount;
    matrixMultiplyChunk_kernel<<<numBlocks, threadsCount>>>(myRank, chunkData, b, chunkResult,
                                                            m, n, k);
    cudaDeviceSynchronize();
}