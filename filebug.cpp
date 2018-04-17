#include <iostream>
#include <fstream>

int main () {
    int n = 10;
    float *buf = new float[n];

    std::string fname = "test.bin";
    std::ifstream file;

    std::remove(fname.c_str());

    // Opening a non-existent file
    file.open(fname, std::ios::in | std::ios::binary);

    std::cout << "file.good(): " << file.good() << std::endl;

    size_t pos = sizeof(float)*0;
    file.seekg(pos, std::ios::beg);
    file.read(reinterpret_cast<char*>(buf),sizeof(float)*n);

    for (int i=0; i<n; i++) {
        std::cout << buf[i] << std::endl;
    }

    return 1;
}

