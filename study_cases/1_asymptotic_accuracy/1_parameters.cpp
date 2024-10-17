#include "1_parameters.hpp"
#include <cmath>

double a = 1;
double b = 1;

double k_x = 2. * M_PI / a;
double k_y = 2. * M_PI / b;

double k_x_squared = k_x * k_x;
double k_y_squared = k_y * k_y;

double U_0 = 0.;

double dirichlet_condition(double m) {
    return U_0;
}

double constant(double x, double y) {
    return U_0;
}

double f(double x, double y) {
    return 2. * (-a * x + x * x - b * y + y * y);
}

double sin_sin(double x, double y) {
    return sin(k_x * x) * sin(k_y * y);
}

double laplacien_sin_sin(double x, double y) {
    return -(k_x_squared + k_y_squared) * sin_sin(x, y);
}

double x_min = 0.;
double y_min = 0.;
double x_max = a;
double y_max = b;

int maxIterations = 2000000;
int n_min = 50;
int n_max = 200;
double epsilon_min = 1e-8;
double epsilon_max = 1e-8;

std::string header = "epsilon; h; error; last_iteration\n";