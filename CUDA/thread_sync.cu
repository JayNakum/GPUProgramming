#include <cuda.h>
#include <cuda_runtime.h>

__global__ void d_sumMatrixTriangle(float* d_inputMat, float* d_resultVec, int n)
{
    int j = threadIdx.x;
    float sum = 0.0;
    for(int i = 0 ; i < j ; i++)
        sum += d_inputMat[i * n + j];
    d_resultVec[j] = sum;

    __syncThreads();

    if (j == N-1)
    {
        sum = 0.0;
        for(i = 0 ; i < n ; i++)
            sum += d_resultVec[i];
        d_resultVec[n] = sum;
    }
}

int main()
{
    int n = 1024;
    int sizeOfMat = n * n;
    int sizeOfVec = n + 1;

    // initialize memory

    cudaMemcpy(d_inputMat, inputMat, sizeOfMat * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_resultVec, resultVec, sizeOfMat * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(1, 1, 1);
    dim3 block(n, 1, 1);
    d_sumMatrixTriangle<<<grid, block>>>(d_inputMat, d_resultVec, n);

    cudaMemcpy(resultVec, d_resultVec, sizeOfVec * sizeof(float), cudaMemcpyDeviceToHost)

    // free memory

}
