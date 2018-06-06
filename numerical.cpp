#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <cstdlib>

int main() {

    std::cout << "Single: " << std::numeric_limits<float>::epsilon() << std::endl;
    std::cout << "Double: " << std::numeric_limits<double>::epsilon() << std::endl;
    std::cout << std::endl;

    float big = 1e+9;
    float sml = 1e-9;
    float nil = 0.0f;

    std::cout << "fabs(big-nil): " << std::fabs(big-nil) << std::endl;
    std::cout << "eps*fmax(big): " << std::numeric_limits<float>::epsilon()*fabs(big) << std::endl;
    std::cout << "fabs <= eps  : " << (std::fabs(big-nil) <= std::numeric_limits<float>::epsilon()*std::fmax(fabs(big),fabs(nil))) << std::endl;
    std::cout << std::endl;

    std::cout << "fabs(sml-nil): " << std::fabs(sml-nil) << std::endl;
    std::cout << "eps*fmax(sml): " << std::numeric_limits<float>::epsilon()*fabs(sml) << std::endl;
    std::cout << "fabs <= eps  : " << (std::fabs(sml-nil) <= std::numeric_limits<float>::epsilon()*std::fmax(fabs(sml),fabs(nil))) << std::endl;
    std::cout << std::endl;

    float denom = big + big;
    float m = 2.0/denom;

    denom = nil + nil;
    float n = 0.0; //2.0/denom;

    denom = n + m;
    float o = 2.0/denom;

    std::cout << "m: " << m << std::endl;
    std::cout << "n: " << n << std::endl;
    std::cout << "o: " << o << std::endl;

    return EXIT_SUCCESS;
}
