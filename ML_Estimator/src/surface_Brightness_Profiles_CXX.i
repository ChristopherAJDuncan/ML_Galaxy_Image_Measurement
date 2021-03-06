%module surface_Brightness_Profiles_CXX
%include "std_vector.i"
%include "std_string.i"

%{
#define SWIG_FILE_WITH_INIT
#include "surface_Brightness_Profiles_CXX.h"
%}

// Instantiate templates used by example                                               
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
}

///------ GAUSSIAN
std::vector<double> cxx_GaussSB(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);			


//---- 1st order derivatives
std::vector<double> cxx_GaussSB_dT(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);				

std::vector<double> cxx_GaussSB_de1(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_de2(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

//--- 2nd order derivatives
std::vector<double> cxx_GaussSB_dde1(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_dde2(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_ddT(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_de1dT(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);				

std::vector<double> cxx_GaussSB_de2dT(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_de1de2(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

//std::vector<double> cxx_GaussSB_dTdF(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);				
