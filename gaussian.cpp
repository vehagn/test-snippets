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

    // Create data
    int nx = 51;
    int ny = 51;
    int r  = 10;

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
    std::cout << "sum: " << sum << std::endl;

    // Create kernel
    int filter_nx = 2;
    int filter_ny = 2;

    double sigma = 1.0;

    double *kernel = new double[filter_nx*filter_ny];

    // Calculate 2D Gaussian kernel
    sum = 0.0;
    for (int j=0; j<filter_ny; j++) {
        for (int i=0; i<filter_nx; i++) {
            kernel[idx2(filter_nx,i,j)] = std::exp(-(1.0*i*i+1.0*j*j)/(2.0*sigma*sigma))/(2.0*M_PI*sigma*sigma);
            if ((i==0) && (j==0)) {
                sum +=  kernel[idx2(filter_nx,i,j)];
            }
            else if ((i==0) || (j==0)) {
                sum += 2*kernel[idx2(filter_nx,i,j)];
            }
            else {
                sum += 4*kernel[idx2(filter_nx,i,j)];
            }
        }
    }

    std::cout << "Raw 2D kernel";
    printData(kernel,filter_nx,filter_ny);
    std::cout << "sum: " << sum << std::endl;

    for (int j=0; j<filter_ny; j++) {
        for (int i=0; i<filter_nx; i++) {
            kernel[idx2(filter_nx,i,j)] /= sum;
        }
    }

    // Print kernel
    std::cout << "Normalised 2D kernel";
    printData(kernel,filter_nx,filter_ny);

    // Apply 2D kernel (no padding)
    int nnx = nx - 2*filter_nx;
    int nny = ny - 2*filter_ny;

    double *newData = new double[nnx*nny];
    std::fill(newData, newData+nnx*nny, 0);

    sum = 0.0;
    for (int j=0; j<nny; j++) {
        for (int i=0; i<nnx; i++) {
            //std::cout << std::endl;
            for (int fj=j-filter_ny+1; fj<j+filter_ny; fj++) {
                for (int fi=i-filter_nx+1; fi<i+filter_nx; fi++) {
                    //std::cout << " (" << fi << "," << fj << ") ";
                    newData[idx2(nnx,i,j)] += kernel[idx2(filter_nx,std::abs(fi-i),std::abs(fj-j))]*data[idx2(nx,fi+filter_nx,fj+filter_ny)];
                }
            }
            sum += newData[idx2(nnx,i,j)];
        }
    }

    std::cout << std::endl;

    double *Gauss2D = new double[nnx*nny];
    std::copy(newData, newData+nnx*nny, Gauss2D);

    // Print 2D filtered data
    //std::cout << "2D filtered data:";
    //printData(newData,nnx,nny);
    std::cout << "2D filtered sum: " << sum << std::endl;

    // Calculate 1D Gaussian kernel
    std::fill(newData, newData+nnx*nny, 0);

    double *kern = new double[filter_nx];
    sum = 0.0;

    for (int i=0; i<filter_nx; i++) {
        kern[i] = std::exp(-(1.0*i*i)/(2.0*sigma*sigma))/(sigma*std::sqrt(2*M_PI));
        sum += kern[i];
        if (i>0) sum += kern[i];
    }
    std::cout << "Raw 1D kernel";
    printData(kern,filter_nx,1);

    std::cout << "sum: " << sum << std::endl;
    for (int i=0; i<filter_nx; i++) {
        kern[i] /= sum;
    }
    std::cout << "Normalised 1D kernel";;
    printData(kern,filter_nx,1);

    for (int j=0; j<nny; j++) {
        for (int i=0; i<nnx; i++) {
            for (int fi=i-filter_nx; fi<=i+filter_nx; fi++) {
                newData[idx2(nnx,i,j)] += kern[std::abs(fi-i)]*data[idx2(nx,fi+filter_nx,j+filter_ny)];
            }
        }
    }

    for (int j=0; j<nny; j++) {
        for (int i=0; i<nnx; i++) {
            data[idx2(nx,i+filter_nx,j+filter_ny)] = newData[idx2(nnx,i,j)];
        }
    }

    //printData(newData,nnx,nny);
    sum = 0.0;
    for (int j=0; j<nny; j++) {
        for (int i=0; i<nnx; i++) {
            for (int fj=j-filter_ny; fj<=j+filter_ny; fj++) {
                newData[idx2(nnx,i,j)] += kern[std::abs(fj-j)]*data[idx2(nx,i+filter_nx,fj+filter_ny)];
            }
            newData[idx2(nnx,i,j)] /= 2.0;
            sum += newData[idx2(nnx,i,j)];
        }
    }

    //std::cout << "1D filtered data:";
    //printData(newData,nnx,nny);
    std::cout << "1D filtered sum: " << sum << std::endl;

    sum = 0.0;
    for (int i=0; i<nnx*nny; i++) {
        newData[i] = std::abs(newData[i] - Gauss2D[i]);
        sum += newData[i];
    }
    //std::cout << "Difference between 1D and 2D";
    //printData(newData,nnx,nny);
    std::cout << "Difference: " << sum << std::endl;


    return EXIT_SUCCESS;
}
