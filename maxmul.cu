#include <stdio.h>
#include <cuda.h>

__global__ void vecmul(float *A, float *B, float *C, int size)
{
    // Row and Column indexes:
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Are they bellow the maximum?
    if (col < size && row < size)
    {
        float result = 0;
        for (int ix = 0; ix < size; ix++)
        {
            result += A[row * size + ix] * B[ix * size + col];
        }
        C[row * size + col] = result;
    }
}

extern "C"
{

    int getThreadNum()
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        // printf("max thread num: %d\n", prop.maxThreadsPerBlock);
        // printf("max grid dimensions: %d, %d, %d)\n",
        //        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        return prop.maxThreadsPerBlock;
    }

    void maxmul(float *A, float *B, float *C, int size)
    {

        int threadNum = getThreadNum();
        int blockNum = (size * size - 0.5) / threadNum + 1;

        int total = size * size;
        // Allocate device memory:
        float *gpu_A;
        float *gpu_B;
        float *gpu_C;
        int msize = total * sizeof(float);
        cudaMalloc((void **)&gpu_A, msize);
        cudaMemcpy(gpu_A, A, msize, cudaMemcpyHostToDevice);
        cudaMalloc((void **)&gpu_B, msize);
        cudaMemcpy(gpu_B, B, msize, cudaMemcpyHostToDevice);
        cudaMalloc((void **)&gpu_C, msize);

        // Blocks & grids:

        dim3 blocks(threadNum, threadNum);
        dim3 grid(blockNum, blockNum);

        // Call the kernel:
        vecmul<<<grid, blocks>>>(gpu_A, gpu_B, gpu_C, size);

        // Get the result Matrix:
        cudaMemcpy(C, gpu_C, msize, cudaMemcpyDeviceToHost);

        // Free device matrices
        cudaFree(gpu_A);
        cudaFree(gpu_B);
        cudaFree(gpu_C);
    }
}