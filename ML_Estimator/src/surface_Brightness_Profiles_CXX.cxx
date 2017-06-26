#include <vector>
#include <cmath>

//Internal defintion of size_t
typedef unsigned int size_t;

//---------- GAUSSIAN Surface Brightness Profiles (Flattened). In all of the below, the arguments list goes as:
// flux (scalar)
// e1 - Component of ellipticty along x,y axis
// e2 - Component of ellipticity along x+y, x-y axis
// size  - Width of gaussian (scales as sigma^2 = size^2*(|e|^2 - 1)
// dx: std::vector of doubles giving the distance from the centroid (of each pixel)
// dy: as dx, in y-direction
// - NOTE: Uses only size, e1, e2 --> NO implementation of mag, shear


//---------- Direct SB profile

std::vector<double> cxx_GaussSB(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy)
{
  const size_t nX(dx.size());
  const size_t nY(dy.size());
  std::vector<double> SB(nX*nY, 0.);
  for (size_t i = 0; i < nX; i++){
    for (size_t j = 0; j < nY; j++){
      SB[i*nX + j] = 0.159154943091895*flux*pow(-pow(e1,2)*pow(size,4) - pow(e2,2)*pow(size,4) + pow(size,4),-0.5)*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2) - 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2) + pow(e2,2) - 1.0)));
    }
  }
  return SB;
}

//---------- First Order Derivatives

std::vector<double> cxx_GaussSB_dT(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy)
{
  const size_t nX(dx.size());
  const size_t nY(dy.size());
  std::vector<double> SB(nX*nY, 0.);
  for (size_t i = 0; i < nX; i++){
    for (size_t j = 0; j < nY; j++){
      SB[i*nX + j] = 0.159154943091895*flux*(2.0*pow(e1,2)*pow(size,3 )+ 2.0*pow(e2,2)*pow(size,3 )- 2.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) - 0.159154943091895*flux*pow((-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4)),(-0.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(size,3)*(pow(e1,2 )+pow(e2,2 )- 1.0));
    }
  }
  return SB;
}

std::vector<double> cxx_GaussSB_de1(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy)
{
  const size_t nX(dx.size());
  const size_t nY(dy.size());
  std::vector<double> SB(nX*nY, 0.);
  for (size_t i = 0; i < nX; i++){
    for (size_t j = 0; j < nY; j++){
      SB[i*nX + j] = 0.159154943091895*e1*flux*pow(size,4)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))+ 0.159154943091895*flux*(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) ;
    }
  }
  return SB;
}


std::vector<double> cxx_GaussSB_de2(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy)
{
  const size_t nX(dx.size());
  const size_t nY(dy.size());
  std::vector<double> SB(nX*nY, 0.);
  for (size_t i = 0; i < nX; i++){
    for (size_t j = 0; j < nY; j++){
      SB[i*nX + j] = 0.159154943091895*e2*flux*pow(size,4)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)));
    }
  }
  return SB;
}
				
//--------------------- 2nd Order Derivatives
//__ Auto Terms

std::vector<double> cxx_GaussSB_dde1(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy)
{
  const size_t nX(dx.size());
  const size_t nY(dy.size());
  std::vector<double> SB(nX*nY, 0.);
  for (size_t i = 0; i < nX; i++){
    for (size_t j = 0; j < nY; j++){
      SB[i*nX + j] = 0.477464829275686*pow(e1,2)*flux*pow(size,8)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-2.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.318309886183791*e1*flux*pow(size,4)*(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(size,4)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)),2)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*(4.0*pow(e1,2)*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,3) )- 2.0*e1*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )- 1.0*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)));
    }
  }
  return SB;
}

std::vector<double> cxx_GaussSB_dde2(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy)
{
  const size_t nX(dx.size());
  const size_t nY(dy.size());
  std::vector<double> SB(nX*nY, 0.);
  for (size_t i = 0; i < nX; i++){
    for (size_t j = 0; j < nY; j++){
      SB[i*nX + j] = 0.477464829275686*pow(e2,2)*flux*pow(size,8)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-2.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.318309886183791*e2*flux*pow(size,4)*(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(size,4)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)),2)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*(4.0*dx[i]*dy[j]*e2/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 4.0*pow(e2,2)*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,3) )- 1.0*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)));
    }
  }
  return SB;
}


std::vector<double> cxx_GaussSB_ddT(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy)
{
  const size_t nX(dx.size());
  const size_t nY(dy.size());
  std::vector<double> SB(nX*nY, 0.);
  for (size_t i = 0; i < nX; i++){
    for (size_t j = 0; j < nY; j++){
      SB[i*nX + j] = 0.159154943091895*flux*(6.0*pow(e1,2)*pow(size,2 )+ 6.0*pow(e2,2)*pow(size,2 )- 6.0*pow(size,2))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*(2.0*pow(e1,2)*pow(size,3 )+ 2.0*pow(e2,2)*pow(size,3 )- 2.0*pow(size,3))*(6.0*pow(e1,2)*pow(size,3 )+ 6.0*pow(e2,2)*pow(size,3 )- 6.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-2.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) - 0.318309886183791*flux*(2.0*pow(e1,2)*pow(size,3 )+ 2.0*pow(e2,2)*pow(size,3 )- 2.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(size,3)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) + 0.477464829275686*flux*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(size,4)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) + 0.159154943091895*flux*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*pow(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2),2)*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(size,6)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2));
    }
  }
  return SB;
}

//__ Cross Terms

std::vector<double> cxx_GaussSB_de1dT(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy)
{
  const size_t nX(dx.size());
  const size_t nY(dy.size());
  std::vector<double> SB(nX*nY, 0.);
  for (size_t i = 0; i < nX; i++){
    for (size_t j = 0; j < nY; j++){
      SB[i*nX + j] = 0.159154943091895*e1*flux*pow(size,4)*(6.0*pow(e1,2)*pow(size,3 )+ 6.0*pow(e2,2)*pow(size,3 )- 6.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-2.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.636619772367581*e1*flux*pow(size,3)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) - 0.159154943091895*e1*flux*size*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(e1,2 )+ pow(e2,2 )- 1.0) + 0.159154943091895*flux*(2.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,3)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )- 1.0*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,3)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*(2.0*pow(e1,2)*pow(size,3 )+ 2.0*pow(e2,2)*pow(size,3 )- 2.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) - 0.159154943091895*flux*(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(size,3)*(pow(e1,2 )+ pow(e2,2 )- 1.0));
    }
  }
  return SB;
}

std::vector<double> cxx_GaussSB_de2dT(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy)
{
  const size_t nX(dx.size());
  const size_t nY(dy.size());
  std::vector<double> SB(nX*nY, 0.);
  for (size_t i = 0; i < nX; i++){
    for (size_t j = 0; j < nY; j++){
      SB[i*nX + j] = 0.159154943091895*e2*flux*pow(size,4)*(6.0*pow(e1,2)*pow(size,3 )+ 6.0*pow(e2,2)*pow(size,3 )- 6.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-2.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.636619772367581*e2*flux*pow(size,3)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) - 0.159154943091895*e2*flux*size*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(e1,2 )+ pow(e2,2 )- 1.0) + 0.159154943091895*flux*(2.0*dx[i]*dy[j]/(pow(size,3)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) + 2.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,3)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*(2.0*pow(e1,2)*pow(size,3 )+ 2.0*pow(e2,2)*pow(size,3 )- 2.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) - 0.159154943091895*flux*(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(size,3)*(pow(e1,2 )+ pow(e2,2 )- 1.0));
    }
  }
  return SB;
}


std::vector<double> cxx_GaussSB_de1de2(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy)
{
  const size_t nX(dx.size());
  const size_t nY(dy.size());
  std::vector<double> SB(nX*nY, 0.);
  for (size_t i = 0; i < nX; i++){
    for (size_t j = 0; j < nY; j++){
      SB[i*nX + j] = 0.477464829275686*e1*e2*flux*pow(size,8)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-2.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*e1*flux*pow(size,4)*(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*e2*flux*pow(size,4)*(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*(2.0*dx[i]*dy[j]*e1/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 4.0*e1*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,3) )- 1.0*e2*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)));
    }
  }
  return SB;
}

std::vector<double> cxx_GaussSB_dTdF(double flux, double e1, double e2, double size, std::vector<double> dx, std::vector<double> dy)
{
  //THIS NEEDS WRITTEN
  const size_t nX(dx.size());
  const size_t nY(dy.size());
  std::vector<double> SB(nX*nY, 0.);
  for (size_t i = 0; i < nX; i++){
    for (size_t j = 0; j < nY; j++){
      SB[i*nX + j] = 0.;
    }
  }
  return SB;
}




