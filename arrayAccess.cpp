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

    return 0;
}
