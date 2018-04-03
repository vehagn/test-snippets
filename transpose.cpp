#include <stdio.h>
#include <iostream>
#include <iomanip>

inline int idx2(int nx, int i, int j) {
    return i + j*nx;
}


int main() {
    int nx = 3;
    int ny = 3;
    int ntot = nx*ny;

    float *A = new float [ntot];
    float *B = new float [ntot];

    for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
            A[idx2(nx,i,j)] = i;
        }
    }

    for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
            B[idx2(ny,j,i)] = A[idx2(nx,i,j)];
            std::cout << idx2(ny,j,i) << "\t" << idx2(nx,i,j) << std::endl;
        }
    }

    for (int i=0; i<nx; i++) {
        std::cout << std::endl;
        for (int j=0; j<ny; j++) {
            std::cout << std::setw(2) << A[idx2(nx,i,j)] << " ";
        }
    }
    std::cout << std::endl;

    for (int i=0; i<nx; i++) {
        std::cout << std::endl;
        for (int j=0; j<ny; j++) {
            std::cout << std::setw(2) << B[idx2(nx,i,j)] << " ";
        }
    }

    std::cout << std::endl;
    for (int i=0; i<ntot; i++) {
        std::cout << B[i] << "\t" << A[i] << std::endl;
    }

    return 0;
}
