#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <fstream>

inline int idx(int w, int h, int i, int j) {
    return i + ((h-1)-j)*w;
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

void readBMP(int *img, std::string _filename) {

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

        // Get file info
        uint32_t fileSize = *reinterpret_cast<uint32_t *>(&header[2]);
        uint32_t offset   = *reinterpret_cast<uint32_t *>(&header[10]);
        uint32_t width    = *reinterpret_cast<uint32_t *>(&header[18]);
        uint32_t height   = *reinterpret_cast<uint32_t *>(&header[22]);
        uint32_t depth    = *reinterpret_cast<uint32_t *>(&header[28]);
        uint32_t compress = *reinterpret_cast<uint32_t *>(&header[30]);

        // Compatability checks
        if (compress !=0) {
            std::cerr << "Warning: Image compression is not supported. Compression type: " << compress << std::endl;
        }
        if (depth != 24) {
            std::cerr << "Warning: Only 24-bit files are supported. This file is " << depth  << "-bit." << std::endl;
        }

        // Takes into account padding.
        uint32_t dataSize = ((width*3 + 3) & (~3))*abs(height);

        // Read from file
        unsigned char *buf = new unsigned char[dataSize];
        file.seekg(offset, std::ios::beg);
        file.read(reinterpret_cast<char*>(buf),dataSize);
        file.close();

        // Read into temporary array. RGB-channels are summed.
        int *tmp = new int[width*height];
        for (int i=0; i<dataSize; i+=3) {
            tmp[i/3] = int(buf[i] & 0xff) + int(buf[i+1] & 0xff) + int(buf[i+2] & 0xff);
        }

        if (height > 0) {
            // Flip image data
            for (int j=0; j<height; j++) {
                for (int i=0; i<width; i++) {
                    img[width*j+i] = tmp[width*(height-1-j)+i];
                }
            }
        }
        else {
            for (int i=0; i<width*height; i++) {
                img[i] = tmp[i];
            }
        }
        delete[] header;
        delete[] buf;
        delete[] tmp;
    }
    else {
        std::cerr << "Error: unable to open file!" << std::endl;
    }
}

int main (int argc, char *argv[]) {
    int w =    20; // 20;
    int h =    10; // 10;
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

    for (int j=0; j<h; j++) {
        std::cout << std::endl;
        for (int i=0; i<w; i++) {
            std::cout << std::setw(3) << pix[idx(w,h,i,j)] << " ";
        }
    }

    std::cout << std::endl;
    writeImg(pix,w,h,"img.bmp");

    int *img = new int[w*h];

    readBMP(img,"img.bmp");

    for (int j=0; j<h; j++) {
        std::cout << std::endl;
        for (int i=0; i<w; i++) {
            std::cout << std::setw(3) << img[i+w*j] << " ";
        }
    }
    std::cout << std::endl;

    return 0;
}
