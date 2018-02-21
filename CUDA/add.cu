#include <iostream>
#include <math.h>

#define NBLOCKS 1024
#define NTHREADS 1024

__global__
void add(int n, float *x, float *y) {
    for (int i=blockIdx.x*blockDim.x + threadIdx.x; i<n; i+=blockDim.x*gridDim.x) {
        y[i] = x[i] + y[i];
    }
}

int main(int argc, char* argv[]) {
    int N = 1<<20;

    float *x, *y;
   
    // Allocate unified memory - GPU and CPU accessible
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialise arrays on host
    for (int i=0; i<N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on N elts on CPU
    int nThreads = 1024;
    int nBlocks  = (N + nThreads - 1)/nThreads;

    std::cout << "Blocks:  " << nBlocks  << std::endl;
    std::cout << "Threads: " << nThreads << std::endl;

    add<<<nBlocks,nThreads>>>(N, x, y);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for errors (All values should be 3.0f;
    float maxErr = 0.0f;
    for (int i=0; i<N; i++) {
        maxErr = fmax(maxErr, fabs(y[i]-3.0f));
    }
    std::cout << "Max err: " << maxErr << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
