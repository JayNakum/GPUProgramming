#include <cuda.h>
#include <cuda_runtime.h>

void printDeviceProps(const cudDeviceProp& devProp, int i)
{
    printf("\nDEVICE %d\n", i);
    printf("Major revision number : %d \n", devProp.major);
    printf("Minor revision number : %d \n", devProp.minor);
    printf("Name : %s\n ", devProp.name);
    printf("Total global memory : u\n", devProp.totalGlobalMem);
    printf("Total shared memory per block :% u\ n", devProp.sharedMemPerBlock);
    printf("Total registers per block : %d \n", devProp.regsPerBlock);
    printf("Warp size : %d\ n", devProp.warpSize);
    printf("Maximum memory pitch : %u\ n", devProp.memPitch);
    printf("Maximum threads per block : %d \n", devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++ i )
        printf("Maximum dimension %d of block : %d \n",i , devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++ i )
        printf("Maximum dimension %d of grid : %d\n ", i , devProp.maxGridSize[i]);
    printf("Clock rate : %d \n", devProp.clockRate);
    printf("Total constant memory :% u\n ", devProp.totalConstMem);
    printf("Texture alignment : %u\ n", devProp.textureAlignment);
    printf("Concurrent copy and execution : %s \n", (devProp.deviceOverlap ? " Yes " : " No "));
    printf("Number of multiprocessors : %d \n", devProp.multiProcessorCount);
}

int main()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    for (int i = 0 ; i < devCount ; ++i)
    {
        cudDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDeviceProps(devProp, i);
    }
}
