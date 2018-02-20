#include <iostream>
#include <math.h>

#define NBLOCKS 1024
#define NTHREADS 1024

__global__
void add(int n, float *x, float *y) {
    for (int i=0; i<n; i++) {
        y[i] = x[i] + y[i];
    }
}

__global__
void gpu_add(int N, float *x, float *y){
    int p;
    for (p=blockIdx.x*blockDim.x + threadIdx.x; p<N; p+=blockDim.x*gridDim.x) {
        y[p] = x[p] + y[p];
    }
}

int main(int argc, char* argv[]) {
    int N = 1<<20;

    float *x = new float[N];
    float *y = new float[N];

    // initialise arrays on host
    for (int i=0; i<N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on N elts on CPU
    add(N, x, y);

    // Check for errors (All values should be 3.0f;
    float maxErr = 0.0f;
    for (int i=0; i<N; i++) {
        maxErr = fmax(maxErr, fabs(y[i]-3.0f));
    }
    std::cout << "Max err: " << maxErr << std::endl;

    // Free memory
    delete[] x;
    delete[] y;

    return 0;
}
