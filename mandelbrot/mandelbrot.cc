#include <complex>
#include <iostream>

bool include_mandelbrot_set(std::complex<float> c) {
    std::complex<float> z(0.0, 0.0);
    for(int i = 0; i < 20; i++) {
        if(abs(z) > 2.0) {
             return false;
        }
        z = z * z + c;
    }
    return true;
}

int main() {
    int nx = 100;
    int ny = 100;
    float xmin = -2.0;
    float ymin = -2.0;
    float xmax =  2.0;
    float ymax =  2.0;
    float dx = (xmax - xmin)/(double)nx;
    float dy = (ymax - ymin)/(double)ny;

    for(int iy = 0; iy < ny; iy++) {
        for(int ix = 0; ix < nx; ix++) {
            float x = xmin + dx*(double)ix;
            float y = ymin + dy*(double)iy;
            std::complex<float> c(x, y);
            if(include_mandelbrot_set(c)) {
                //std::cout << x << "  " << y << std::endl;
                std::cout << "*";
            }else {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
}