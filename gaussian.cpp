#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <cstdlib>

inline int idx2(int nx, int i, int j) {
    return i + j*nx;
}

inline void printData(double *data, int nx, int ny) {
    for (int j=0; j<ny; j++) {
        std::cout << std::endl;
        for (int i=0; i<nx; i++) {
            std::cout << std::setw(10) << data[idx2(nx,i,j)] << " ";
        }
    }
    std::cout << std::endl;
}

int main() {
    std::cout.precision(3);
    std::fixed;

    // Kernel size
    int filter_nx = 10;
    int filter_ny = 10;

    double sigma = 1.0;

    // Create data
    int nx = 501;
    int ny = 501;
    int r  =  10;

    double *data = new double[nx*ny];
    double sum = 0.0;

    for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
            if (((i-nx/2)*(i-nx/2) + (j-ny/2)*(j-ny/2))<r*r) {
                data[idx2(nx,i,j)] = 1.0;
            }
            else {
                data[idx2(nx,i,j)] = 0.0;
            }
            sum += data[idx2(nx,i,j)];
        }
    }

    // Print data
    //printData(data,nx,ny);
    std::cout << "Data sum: " << sum << std::endl;

    // Create 2D kernel
    double *kernel2D = new double[filter_nx*filter_ny];

    // Calculate 2D Gaussian kernel
    sum = 0.0;
    for (int j=0; j<filter_ny; j++) {
        for (int i=0; i<filter_nx; i++) {
            kernel2D[idx2(filter_nx,i,j)] = std::exp(-(1.0*i*i+1.0*j*j)/(2.0*sigma*sigma))/(2.0*M_PI*sigma*sigma);
            if ((i==0) && (j==0)) {
                sum +=  kernel2D[idx2(filter_nx,i,j)];
            }
            else if ((i==0) || (j==0)) {
                sum += 2*kernel2D[idx2(filter_nx,i,j)];
            }
            else {
                sum += 4*kernel2D[idx2(filter_nx,i,j)];
            }
        }
    }

    //std::cout << "Raw 2D kernel";
    //printData(kernel,filter_nx,filter_ny);
    std::cout << "2D kernel sum: " << sum << std::endl;

    //Normalise kernel
    for (int j=0; j<filter_ny; j++) {
        for (int i=0; i<filter_nx; i++) {
            kernel2D[idx2(filter_nx,i,j)] /= sum;
        }
    }

    // Print kernel
    //std::cout << "Normalised 2D kernel";
    //printData(kernel,filter_nx,filter_ny);

    // Apply 2D kernel (no padding)
    int nnx = nx - 2*filter_nx;
    int nny = ny - 2*filter_ny;

    double *Gauss2D = new double[nnx*nny];
    std::fill(Gauss2D, Gauss2D+nnx*nny, 0);

    sum = 0.0;
    for (int j=0; j<nny; j++) {
        for (int i=0; i<nnx; i++) {
            for (int fj=j-filter_ny+1; fj<j+filter_ny; fj++) {
                for (int fi=i-filter_nx+1; fi<i+filter_nx; fi++) {
                    Gauss2D[idx2(nnx,i,j)] += kernel2D[idx2(filter_nx,std::abs(fi-i),std::abs(fj-j))]*data[idx2(nx,fi+filter_nx,fj+filter_ny)];
                }
            }
            sum += Gauss2D[idx2(nnx,i,j)];
        }
    }

    // Print 2D filtered data
    //std::cout << "2D filtered data:";
    //printData(Gauss2D,nnx,nny);
    std::cout << "2D filtered sum: " << sum << std::endl;

    // Calculate 1D Gaussian kernel
    double *kernel1D = new double[filter_nx];
    sum = 0.0;

    for (int i=0; i<filter_nx; i++) {
        kernel1D[i] = std::exp(-(1.0*i*i)/(2.0*sigma*sigma))/(sigma*std::sqrt(2*M_PI));
        sum += kernel1D[i];
        if (i>0) sum += kernel1D[i];
    }
    //std::cout << "Raw 1D kernel";
    //printData(kernel1D,filter_nx,1);

    std::cout << "1D kernel sum: " << sum << std::endl;
    for (int i=0; i<filter_nx; i++) {
        kernel1D[i] /= sum;
    }
    //std::cout << "Normalised 1D kernel";;
    //printData(kernel1D,filter_nx,1);

    double *Gauss1DX = new double[nnx*nny];
    std::fill(Gauss1DX, Gauss1DX+nnx*nny, 0);

    sum = 0.0;
    for (int j=0; j<nny; j++) {
        for (int i=0; i<nnx; i++) {
            for (int fi=i-filter_nx+1; fi<i+filter_nx; fi++) {
                Gauss1DX[idx2(nnx,i,j)] += kernel1D[std::abs(fi-i)]*data[idx2(nx,fi+filter_nx,j+filter_ny)];
            }
            sum += Gauss1DX[idx2(nnx,i,j)];
        }
    }
    std::cout << "1DX filtered sum: " << sum << std::endl;

    double *buf = new double[nx*ny];
    std::fill(buf, buf+nx*ny, 0);
    for (int j=0; j<nny; j++) {
        for (int i=0; i<nnx; i++) {
            buf[idx2(nx,i+filter_nx,j+filter_ny)] = Gauss1DX[idx2(nnx,i,j)];
        }
    }

    double *Gauss1DXY = new double[nnx*nny];
    std::fill(Gauss1DXY, Gauss1DXY+nnx*nny, 0);
    sum = 0.0;
    for (int i=0; i<nnx; i++) {
        for (int j=0; j<nny; j++) {
            for (int fj=j-filter_ny+1; fj<j+filter_ny; fj++) {
                Gauss1DXY[idx2(nnx,i,j)] += kernel1D[std::abs(fj-j)]*buf[idx2(nx,i+filter_nx,fj+filter_ny)];
            }
            sum += Gauss1DXY[idx2(nnx,i,j)];
        }
    }
    //std::cout << "1D filtered data:";
    //printData(Gauss1DXY,nnx,nny);

    std::cout << "1DXY filtered sum: " << sum << std::endl;

    double *diff = new double[nnx*nny];
    double maxAbsDiff = 0;

    sum = 0.0;
    for (int i=0; i<nnx*nny; i++) {
        diff[i] = std::abs(Gauss1DXY[i] - Gauss2D[i]);
        maxAbsDiff = (diff[i]>maxAbsDiff)?(diff[i]):(maxAbsDiff);
        sum += diff[i];
    }
    //std::cout << "Difference between 1D and 2D";
    //printData(Gauss1DXY,nnx,nny)
    std::cout << "Maximum absolute difference:    " << maxAbsDiff << std::endl;
    std::cout << "Absolute cumulative difference: " << sum << std::endl;

    delete[] data;

    delete[] kernel2D;
    delete[] Gauss2D;

    delete[] kernel1D;
    delete[] Gauss1DX;
    delete[] Gauss1DXY;
    delete[] buf;
    delete[] diff;

    return EXIT_SUCCESS;
}
