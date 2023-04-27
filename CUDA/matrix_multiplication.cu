#include <cuda.h>
#include <cuda_runtime.h>

__global__ void multiplyMat(float* d_matrixM, float* d_matrixN, float* d_resultMatrix, int n)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if ((i < n) && (j < n))
    {
        float result = 0.0;
        for (int k = 0 ; k < n ; k++)
        {
            result += d_matrixM[i * n + k] * d_matrixN[k * n + j];
        }
        d_resultMatrix[i*N + j] = result;
    }
}

int main()
{
    int size = 16 * 16;
    cudaMemcpy(d_matrixM, matrixM, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixN, matrixN, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(2, 2, 1);
    dim3 block(8, 8, 1);

    int n = 16;
    multiplyMat<<<grid, block>>>(d_matrixM, d_matrixN, d_resultMatrix, n);

    cudaMemcpy(resultMatrix, d_resultMatrix, size*sizeof(float), cudaMemcpyDeviceToHost);
}