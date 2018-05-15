#include <stdio.h>
#include <iostream>

int main() {
    int a[] = {1,2,3,4,5,6,7,8,9,10};
    printf("%d %d %d\n", a[2], 7[a], (3+4)[a-2]);

    float A = 2. ,B = 0.0;
    std::cout << A << " " << B << std::endl;

    int nx = 5;
    int ny = 5;
    int ntot = nx*ny;
    float* M = new float[ntot];

    for (int i=0; i<ntot; i++) {
        M[i] = i<<1;
    }

    for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
            std::cout <<  M[i+nx*j] << " ";
        }
    }
    std::cout << std::endl;

    std::cout << "Incrementation: " << std::endl;
    int p = 0;
    int q = 0;
    for (int i=0; i<5; i++) {
        std::cout << "++p " << ++p << std::endl;
        std::cout << "q++ " << q++ << std::endl;
    }

    std::cout << "p   " << p << std::endl;
    std::cout << "q   " << q << std::endl;

    int b[6] = {0,0,0,0,0,0};

    std::cout << b[2] << std::endl;

    std::cout << "Pointers to pointers: " << std::endl;
    float *k = new float[4];
    float *l = new float[4];
    float *m = new float[4];

    for (int i=0; i<4; i++) {
        k[i] = 1;
        l[i] = 2;
        m[i] = 3;
    }

    float **H = new float*[3]{k,l,m};
    float *n = H[0];

    for (int i=0; i<4; i++) {
        std::cout << "n[i]:    " << n[i] << std::endl;
        std::cout << "H[1][i]: " << H[1][i] << std::endl;
    }

    return EXIT_SUCCESS;
}
