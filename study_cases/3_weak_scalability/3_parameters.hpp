#ifndef THREE_PARAMETERS_H
#define THREE_PARAMETERS_H

#include <iostream>
#include <cmath>

extern double a;
extern double b;

extern double k_x;
extern double k_y;
extern double k_x_squared;
extern double k_y_squared;
extern double U_0;

double dirichlet_condition(double m);
double constant(double x, double y);
double f(double x, double y);
double sin_sin(double x, double y);
double laplacien_sin_sin(double x, double y);

extern double x_min;
extern double y_min;
extern double x_max;
extern double y_max;

extern int Nx;
extern int Ny;

extern int max_iterations;
extern double epsilon;

extern std::string header;
extern std::string folder;
extern std::string file_name_jacobi;
extern std::string file_name_gauss_seidel;

#endif // THREE_PARAMETERS_H