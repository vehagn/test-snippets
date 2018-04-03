#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <fstream>

inline int idx(int w, int h, int i, int j) {
    return i + ((h-1)-j)*w;
}

inline int idx2(int nx, int i, int j) {
    return i + j*nx;
}

void writeImg(const int *red, const int *blue, const int *green, const int w, const int h, const char* fname) {
    // Create and save a .bmp image file using red, blue, green channels
    int filesize = 54 + 3*w*h;
    unsigned char *img = new unsigned char[3*w*h];
    int ind = 0;

    for (int i=0; i<w; i++) {
        for (int j=0; j<h; j++) {
            ind = idx(w,h,i,j);
            img[ind*3+2] = (unsigned char)(red[ind]  ); // r
            img[ind*3+1] = (unsigned char)(blue[ind] ); // g
            img[ind*3+0] = (unsigned char)(green[ind]); // b
        }
    }

    // Magic header mumbo-jumbo
    unsigned char bmpfileheader[14] = {'B','M',0,0,0,0,0,0,0,0,54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0,0,0,0,0,0,0,0,0,1,0,24,0};
    unsigned char bmppad[3] = {0,0,0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(       w    );
    bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
    bmpinfoheader[ 6] = (unsigned char)(       w>>16);
    bmpinfoheader[ 7] = (unsigned char)(       w>>24);
    bmpinfoheader[ 8] = (unsigned char)(       h    );
    bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
    bmpinfoheader[10] = (unsigned char)(       h>>16);
    bmpinfoheader[11] = (unsigned char)(       h>>24);

    FILE *f;

    f = fopen(fname,"wb");
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    for (int i=0; i<h; i++) {
        fwrite(img+(w*(h-i-1)*3),3,w,f);
        fwrite(bmppad,1,(4-(w*3)%4)%4,f);
    }

    free(img);
    fclose(f);
}

void writeImg(const int *grey, const int w, const int h, const char* fname) {
    // Create and save a .bmp file in greyscale calling the rgb-version.
    writeImg(grey, grey, grey, w, h, fname);
}

void readBMP(int *img, int ntot, std::string _filename) {

    std::ifstream file;
    file.open(_filename, std::ios::in | std::ios::binary);

    if (file.is_open()) {
        // Create array to store header
        static constexpr size_t BMPHEADER_SIZE = 54;
        char *header = new char[BMPHEADER_SIZE];
        file.read(header,BMPHEADER_SIZE);

        // Check that we have a bmp-file by looking for markers in the header.
        if ((header[0] != 'B') || (header[1] != 'M')){
            std::cerr << "Error: BMP header not detected!" << std::endl;
            return;
        }

        // Get file info from header
        int32_t fileSize = *reinterpret_cast<int32_t *>(&header[2]);
        int32_t offset   = *reinterpret_cast<int32_t *>(&header[10]);
        int32_t width    = *reinterpret_cast<int32_t *>(&header[18]);
        int32_t height   = *reinterpret_cast<int32_t *>(&header[22]);
        int32_t depth    = *reinterpret_cast<int32_t *>(&header[28]);
        int32_t compress = *reinterpret_cast<int32_t *>(&header[30]);

        // Compatability checks
        bool invHeight = false;
        if (height < 0) {
            // Someone decided inverse height should be allowed in the BMP standard, so we have to take account of this
            invHeight = true;
            height = -height;
        }
        if (ntot != width*height){
            std::cerr << "Error: Can't cast image of width " << width << " and height " << height << " into array of size " << ntot << std::endl;
        }
        if (compress !=0) {
            std::cerr << "Warning: Image compression is not supported. Compression type: " << compress << std::endl;
        }
        if (depth != 24) {
            std::cerr << "Warning: Only 24-bit files have been tested. This file is " << depth  << "-bit." << std::endl;
        }

        // Takes into account padding.
        int32_t rowSize = ((width*3 + 3) & (~3));
        int32_t dataSize = rowSize*height;

        // Read from file
        char *buf = new char[dataSize];
        file.seekg(offset, std::ios::beg);
        file.read(reinterpret_cast<char*>(buf),dataSize);
        file.close();

        // Read into image array.
        // Note that rowSize can be larger than width of image due to padding.
        for (int j=0; j<height; j++) {
            for (int i=0; i<3*width; i+=3) {
                // Make sure we read data in right direction.
                int ind = (invHeight)?(idx2(width,i/3,j)):(idx2(width,i/3,height-1-j));
                // The data we read is 8 bit which can show values between 0 and 255, or 0x00 and 0xff in hex.
                img[ind] = (int(buf[idx2(rowSize,i+0,j)] & 0xff) + int(buf[idx2(rowSize,i+1,j)] & 0xff) + int(buf[idx2(rowSize,i+2,j)] & 0xff))/3;
            }
        }
        delete[] header;
        delete[] buf;
    }
    else {
        std::cerr << "Error: unable to open file!" << std::endl;
    }
}

int main (int argc, char *argv[]) {
    int w =  1024; // 64;
    int h =   192; // 12;
    int ind =   0;

    int *pix = new int[w*h];

    // Create an image with some features
    for (int i=0; i<w; i++) {
        for (int j=0; j<h; j++) {
            ind = idx(w,h,i,j);
            pix[ind] = 255; // Make sure we fill a color;
            if (   i         < w/4) pix[ind] = 127;
            if (     j       < h/4) pix[ind] =  63;
            if (   i+j - h/4 < h  ) pix[ind] =  31;
            if (-2*i+j + 4*h < w  ) pix[ind] =   0;
        }
    }

    //for (int j=0; j<h; j++) {
    //    std::cout << std::endl;
    //    for (int i=0; i<w; i++) {
    //        std::cout << std::setw(3) << pix[idx(w,h,i,j)] << " ";
    //    }
    //}
    //std::cout << std::endl;

    //writeImg(pix,w,h,"img.bmp");

    int *img = new int[w*h];

    readBMP(img,w*h,"ntnu1024x192.bmp");

    //for (int j=0; j<h; j++) {
    //    std::cout << std::endl;
    //    for (int i=0; i<w; i++) {
    //        std::cout << std::setw(3) << img[i+w*j] << " ";
    //    }
    //}
    //std::cout << std::endl;

    writeImg(img,w,h,"out.bmp");

    return 0;
}
