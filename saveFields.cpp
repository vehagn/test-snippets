#include <vector>
#include <iostream>

// Create bit masks for each of the fields
constexpr unsigned int SXX = 0b000000001;
constexpr unsigned int SYY = 0b000000010;
constexpr unsigned int SZZ = 0b000000100;
constexpr unsigned int SXY = 0b000001000;
constexpr unsigned int SXZ = 0b000010000;
constexpr unsigned int SYZ = 0b000100000;
constexpr unsigned int VX  = 0b001000000;
constexpr unsigned int VY  = 0b010000000;
constexpr unsigned int VZ  = 0b100000000;

int main (int argc, char* argv[]) {

    int nx = 5;

    // Initialise all fields
    float* sxx = new float[nx];
    float* syy = new float[nx];
    float* szz = nullptr;
    float* sxy = new float[nx];
    float* sxz = nullptr;
    float* syz = nullptr;
    float* vx  = new float[nx];
    float* vy  = new float[nx];
    float* vz  = nullptr; 

    // Fill fields with some data
    for (int i=0; i<nx; i++) {
        sxx[i] = 1;
        syy[i] = 2;
        sxy[i] = 3;
        vx[i]  = 4;
        vy[i]  = 5;
    }

    unsigned int Fields;
    std::vector<float*> saveFields;

    Fields = VX|SXX|SZZ|SYY; // Assume this is input. Order does not matter.

    // Add an element in the vector of which fields should be saved.
    // Fields will always be stored in the sequence here, 
    // i.e. SXX -> SYY -> SZZ ->...
    if ((Fields & SXX) && (sxx != nullptr)) saveFields.push_back(sxx);
    if ((Fields & SYY) && (syy != nullptr)) saveFields.push_back(syy);
    if ((Fields & SZZ) && (szz != nullptr)) saveFields.push_back(szz);
    if ((Fields & SXY) && (sxy != nullptr)) saveFields.push_back(sxy);
    if ((Fields & SYZ) && (syz != nullptr)) saveFields.push_back(syz);
    if ((Fields & VX ) && (vx  != nullptr)) saveFields.push_back(vx);
    if ((Fields & VY ) && (vy  != nullptr)) saveFields.push_back(vy);
    if ((Fields & VZ ) && (vz  != nullptr)) saveFields.push_back(vz);

    int n_buf = nx*saveFields.size();
    float* buffer = new float[n_buf];

    // Copying all selected wavefields to buffer
    std::cout << "Size of saveFields: " << saveFields.size() << std::endl;
    for (int v=0; v<saveFields.size(); v++) {
        for (int i=0; i<nx; i++) {
            buffer[i+v*nx] = saveFields.at(v)[i];
        }
    }

    // Printing our buffer
    std::cout << "Writing out full buffer: " << std::endl;
    for (int b=0; b<n_buf; b++) {
        std::cout << buffer[b] << std::endl;
    }
}
