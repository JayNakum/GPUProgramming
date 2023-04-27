#include <cuda.h>
#include <cuda_runtime.h>

__global__ void d_vectorAdd(float* vecA, float* vecB, float* resultVec, int n)
{
    // CUDA kernel definition
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i < n)
        resultVec[i] = vecA[i] + vecB[i];
}

// host program
void h_vectorAdd(float* h_vecA, float* h_vecB, float* h_resultVec, int n)
{
    int size = n * sizeof(float);
    float *d_vecA = NULL, *d_vecB = NULL, *d_resultVec = NULL;
    cudaError_t err = cudaSuccess;

    printf("Allocating Memory.\n");

    err = cudaMalloc((void **)&d_vecA, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(%s) Failed to allocate device vector d_vecA.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_vecB, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(%s) Failed to allocate device vector d_vecB.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void **)&d_resultVec, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(%s) Failed to allocate device vector d_resultVec.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copying input data from host to device.\n");
    
    err = cudaMemcpy(d_vecA, h_vecA, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(%s) Failed to copy vecA.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_vecA, h_vecB, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(%s) Failed to copy vecB.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock, -1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads.\n", blocksPerGrid, threadsPerBlock);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_vecA, d_vecB, d_resultVec, n);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(%s) Failed to launch vectorAdd() kernel.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copying output data from device to host.\n");

    err = cudaMemcpy(h_resultVec, d_resultVec, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(%s) Failed to copy resultVec.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_vecA);
    cudaFree(d_vecB);
    cudaFree(d_resultVec);

    // Result Verification
    for(int i = 0 ; i < n ; ++i)
    {
        if(fabs(h_vecA[i] + h_vecB[i] - h_resultVec[i]) > 1e-5)
        {
            printf("Test FAILED\n");
            fprintf(stderr, "Result verification failed at element %d.\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");
}
