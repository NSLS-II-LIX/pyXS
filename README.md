# pyXS

This free software was initially written by Lin Yang for users at beamline X9 of NSLS. Use at your own risk.
Thanks to Kevin Yager and Ian Berke for suggestions and bug fixes.

(1) This package depends on the following software packages:

	python
	numpy
	python image library (PIL)
	matplotlib
	scipy (for line profile extraction)
	SWIG (for compiling the RQconv module)

(2) This package also include a C program that needs to be compiled for your particular OS.
Simply type the following under the directory pyXS:

	python setup.py build_ext --inplace

or using mingw under windows:

	python setup.py build_ext --inplace --compile=mingw32

This should generate a new file _RQconv.so (_RQconv.pyd under Windows).

NOTE: compiling using gcc-4 might not work under windows (python is compiled using gcc-3?)
The solution is to edit Lib\distutils\cygwinccompiler.py under the python directory
and remove the -mno-cygwin flag manually.

(3) Older versions of PIL have problem correctly reading TIFF files that have Big Endian byte
order. You will know if your scattering data appear scrambled. Update PIL.


The pyXS package include the following modules:

## Data2D

- This module reads (FabIO) and displays (matplotlib) 2D scattering patterns and perform the conversion between pixel position and reciprocal space coordinates, q or (qr, qz).
- Data2D relies on C code RQconv.c to convert data, based on scattering geometry defined by the structure ExpPara.
- The line profile extraction and annotation functions of view.gtk can be similarly accomplished using functions defined in this module.

## slnXS

- This module is used to process isotropic scattering data (e.g. solution scattering).
- It can average from multiple scattering patterns, perform background subtraction (dark current and buffer scattering) based on transmitted beam intensity, and combine SAXS and WAXS data simultaneously collected on the two detectors.
- Guinier plot and approximate p(r) function calculation.





