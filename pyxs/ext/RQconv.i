%module(package="pyxs") RQconv
%{
#include <math.h>
#include <sys/types.h>
#define SWIG_FILE_WITH_INIT
#include "RQconv.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* grid, int nc, int len)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* data, int row, int col)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* data, int nrow, int ncol)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* mask, int mrow, int mcol)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* dataout, int drow, int dcol)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* data1, int row1, int col1), (int* data2, int row2, int col2)}

%include "RQconv.h"
