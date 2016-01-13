//#include <sys/types.h>

#ifndef Pi
#define Pi 3.1415926
#endif

typedef struct {
    float wavelength;
    float  bm_ctr_x;
    float  bm_ctr_y;
    // ratioDw: sample_to_detector distance / detector_physical_width
    float ratioDw;
    float det_orient;
    float det_tilt;
    float det_phi;
    int grazing_incident;                   // True/False
    float incident_angle;
    float sample_normal;
    float beam_tX, beam_tY;
    int flip;                               // True/False, True for PSI WAXS
    float det_rot[3][3], det_rot_T[3][3];    // the rotation matrix
    float qr0, qz0, dq;
    int nr, nz;
} ExpPara;

typedef struct {
  float x,y;
} vec2;

// determines whether the pixel is a zinger
int zinger(int *data, int row, int col, int x, int y, int box, float tol);

// scattering data are int  
void dezinger(int *data, int row, int col, int *mask, int mrow, int mcol, int box, float tol); 

// returns the pixel position (x,y) corresponding to (qr, qz)
vec2 qrqz2xy(int *data, int row, int col, ExpPara *exp, float x, float y, int flag);

vec2 qphi2xy(int *data, int row, int col, ExpPara *exp, float x, float y);

// returns q
float xy2q(int *data, int row, int col, ExpPara *exp, float x, float y); 

// this will return a (qr, qz) pair
vec2 xy2qrqz(int *data, int row, int col, ExpPara *exp, float x, float y);

// useful part of xy2q and xy2qrqz
float pos2qrqz(int *data, int row, int col, ExpPara *exp, float x, float y, float *pqr, float *pqz, float *ic);

// claculate the correction factor for incident angle dependence 
void cor_IAdep_2D(int *data, int row, int col, ExpPara *exp, int corCode, int invert);

// convert 2D solution scattering/powder diffraction data into 1D curve 
//float *conv_to_Iq(ExpPara *exp, int *data, int w, int h);
void conv_to_Iq(int* data, int row, int col, ExpPara *exp, double *grid, int nc, int len, int cor);

// simply return the interpolated intensity at non-integer position (fx,fy)
float get_value(int *data, int nrow, int ncol, float fx, float fy);

// this is needed to figure out how big the output (reciprocal space) 2D data should be
void pre_conv_Iqrqz(int *data, int nrow, int ncol, ExpPara *exp);

//
void conv_to_Iqrqz(int *data, int row, int col, int *dataout, int drow, int dcol, ExpPara *exp);

// merge 2 images together, using the "dezinger" algorithm
void merge(int *data1, int row1, int col1, int *data2, int row2, int col2);

void calc_rot(ExpPara *exp);
