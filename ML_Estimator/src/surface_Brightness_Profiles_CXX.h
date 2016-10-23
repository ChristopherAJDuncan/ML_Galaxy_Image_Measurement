#include <vector>
#include <cmath>

std::vector<double> cxx_GaussSB(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_dT(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);				

std::vector<double> cxx_GaussSB_de1(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_de2(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_dde1(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_dde2(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_ddT(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_de1dT(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);				

std::vector<double> cxx_GaussSB_de2dT(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_de1de2(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);

std::vector<double> cxx_GaussSB_dTdF(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy);				
