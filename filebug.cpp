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

    std::cout << "file.is_open(): " << file.is_open() << std::endl;
    std::cout << "file.fail():    " << file.fail()    << std::endl;
    std::cout << "file.good():    " << file.good()    << std::endl;
    std::cout << "file.bad():     " << file.bad()     << std::endl;
    std::cout << "file.eof():     " << file.eof()     << std::endl;

    size_t pos = sizeof(float)*0;
    file.seekg(pos, std::ios::beg);
    file.read(reinterpret_cast<char*>(buf),sizeof(float)*n);

    for (int i=0; i<n; i++) {
        std::cout << buf[i] << std::endl;
    }

    return 1;
}

