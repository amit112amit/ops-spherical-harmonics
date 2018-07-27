#ifndef __SHTOOLS_HEADER__
#define __SHTOOLS_HEADER__

extern "C" void shexpandlsq_wrapper_(double* cilm, double* d, double* lat,
	double* lon, int* N, int* lmax, double* chi2);

extern "C" void makegridpoints_wrapper_(double* cilm, int* lmax, int* n,
	double* lat, double* lon, double* points, int* dealloc);

extern "C" void glqgridcoord_wrapper_(double* latglq, double* longlq,
	int* lmax, int* nlat, int* nlong);

extern "C" void shglq_wrapper_(int* lmax, double* zero, double* w,
	double* plx);

extern "C" void shexpandglq_wrapper_(double* cilm, int* lmax, double* gridglq,
	double* w, double* plx);

extern "C" void makegridglq_wrapper_(double* gridglq, double* cilm, int* lmax,
	double* plx);

extern "C" void shexpanddh_wrapper_(double* grid, int* n, double* cilm,
	int* lmax);

extern "C" void makegriddh_wrapper_(double* grid, int* n, double* cilm,
	int* lmax);

extern "C" void shpowerspectrum_wrapper_(double* cilm, int* lmax,
	double* pspectrum);

extern "C" void plmon_wrapper_(double* p, int* lmax, double* z);

#endif /* ifndef __SHTOOLS_HEADER__ */
