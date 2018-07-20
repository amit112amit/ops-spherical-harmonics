#include <ctime>
#include <vector>
#include <iostream>
#include <random>
#include <math.h>

extern "C" void shexpandlsq_wrapper_(
                double* cilm, double* d, double* lat, double* lon,
                int* nmax, int* lmax, int* norm, double* chi2,
                int* csphase, int* exitstatus);

extern "C" double makegridpoint_wrapper_(
                double* cilm, int* lmax, double* lat, double* lon,
                int* norm, int* csphase, int* dealloc);

int main(){
    int nmax = 72, lmax = 7, norm = 4, csphase = 1, exitstatus = 0, dealloc = 0;
    double chi2 = 0;
    std::vector<double> d(nmax), lat(nmax), lon(nmax), cilm((lmax+1)*(lmax+1)*2, 0.0);
    for( auto i=0; i < nmax; ++i ){
        auto x = 2.0*std::rand() - 1.0;
        auto y = 2.0*std::rand() - 1.0;
        auto z = 2.0*std::rand() - 1.0;
        lat[i] = std::atan2(z, std::sqrt(x*x + y*y))*180.0/M_PI;
        lon[i] = std::atan2(y, x)*180.0/M_PI;
        d[i] = 1.0;
    }

    clock_t t;
    t = clock();
    for(auto j=0; j < 1000; ++j){
        shexpandlsq_wrapper_( cilm.data(), d.data(), lat.data(), lon.data(),
                  &nmax, &lmax, &norm, &chi2, &csphase, &exitstatus);
    }
    t = clock() - t;
    std::cout<< "Time elapsed = " << float(t)/CLOCKS_PER_SEC << std::endl;

    // Now print A_lm where l = 0 to lmax and m = -l to l
    std::cout<< "l\tm\tC_lm\tS_lm\tA_lm"<< std::endl;
    for( auto l = 0; l < lmax+1; ++l){
        for( auto m = -l; m < l+1; ++m){
            auto j = m >= 0? 0 : 1;
            auto n = abs(m);
            auto index_common = 2*(l + (lmax + 1)*n);
            auto index = j + index_common;
            std::cout<< l << "\t" << m << "\t" << cilm[index_common]
                        << "\t" << cilm[index_common + 1]
                        << "\t" << cilm[index] << std::endl;
        }
    }

    // Now let's reconstruct f(theta,phi)
    std::cout<< "Re-construct the input value from the output A_lm" << std::endl;
    for( auto i = 0; i < nmax; ++i){
        auto val = makegridpoint_wrapper_(cilm.data(), &lmax, &lat[i], &lon[i],
                                          &norm, &csphase, &dealloc);
        std::cout<< val << std::endl;
    }
    return 0;
}
