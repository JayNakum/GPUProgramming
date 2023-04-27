#include <cuda.h>
#include <cuda_runtime.h>

struct cuComplex {
    float r;
    float i;
};

__device__ float magnitude(struct cuComplex a)
{
    return (( a .r * a. r) + (a. i * a.i ));
}

__device__ void add(struct cuComplex a , struct cuComplex b , struct cuComplex* res)
{
    res -> r = a.r + b.r;
    res -> i = a.i + b.i;
}
__device__ void mul(struct cuComplex a , struct cuComplex b , struct cuComplex* res)
{
    res -> r= ( a.r * b .r) - ( a.i * b .i);
    res -> i= ( a.r * b .i) + ( a.i * b .r);
}

__device__ int julia ( int x , int y )
{
    const float scale = 1.5;
    float jx = scale * (float) (DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float) (DIM / 2 - y) / (DIM / 2);
    
    struct cuComplex c, a, r1, r2;
    c.r = -0.8; 
    c.i = 0.154;
    a.r = jx;
    a.i = jy;
    
    int i = 0;
    for (i = 0 ; i < 200 ; i++) 
    {
        // a = a* a + c;
        mul(a, a, &r1 );
        add(r1, c, &r2 ) ;
        if (magnitude(r2) > 1000) return 0; // return 0 if it is not in set
        a.r = r2.r;
        a.i = r2.i;
    }
    return 1; // return 1 if point is in set
}

__global__ void kernel(unsigned char* ptr) 
{
    // map from threadIdx / BlockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    int juliaValue = julia(x, y);
    ptr[offset * 4 + 0] = 255 * juliaValue ; // red if 1 , black if 0
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main () 
{
    CPUBitmap bitmap(DIM, DIM);
    unsigned char * dev_bitmap;
    
    cudaMalloc ((void **) &dev_bitmap, bitmap.image_size());
    dim3 grid (DIM, DIM);

    kernel<<<grid, 1>>>(dev_bitmap);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    bitmap.display_and_exit();
    cudaFree(dev_bitmap);
}
