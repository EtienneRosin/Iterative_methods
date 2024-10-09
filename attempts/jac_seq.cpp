#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <vector>
#include <array>

int n = 100;
int Nx = n;
int Ny = n;

const int num_points = (Nx + 2) * (Ny + 2);

double a = 2.;
double b = 2.;
double U_0 = 0.;

double dx = a / (Nx + 1);
double dy = b / (Ny + 1);

double dx2 = dx * dx;
double dy2 = dy * dy;

double A_ii = -2. * (1. / dx2 + 1. / dy2);

// double alpha = 0.5;

double k_x = 2. * M_PI / a;
double k_y = 2. * M_PI / b;

double k_x_squared = k_x*k_x;
double k_y_squared = k_y*k_y;



// std::vector sol;
// std::array<double, num_points> sol;
std::vector<double> sol(num_points, 0.0);
std::cout << sol.size() << "\n";
std::vector<double> new_sol(num_points, 0.0);
std::vector<double> exact_sol(num_points, 0.0);
std::vector<double> source(num_points, 0.0);

double r_init = 0.0;
double r_ = 0.0;

int I(int i, int j){
    return i + (Nx + 2) * j;
};


float ComputeSquaredResidual(const std::vector<double> &u, const std::vector<double> &source){
    double value = 0.0;
    // double tmp = 0.0;
    for (int i = 1; i <= Nx; i++)
    {
        for (int j = 1; j <= Ny; j++){
            double tmp = source[I(i, j)] - ( (u[I(i+1,j)] - 2.*u[I(i,j)] + u[I(i-1,j)])/dx2 + (u[I(i,j+1)] - 2.*u[I(i,j)] + u[I(i,j-1)])/dy2);
            value += tmp * tmp;
        }
    }
    return value;
};

float ComputeSquaredError(const std::vector<double> &u, const std::vector<double> &u_exact){
    double value = 0.0;
    // double tmp = 0.0;
    for (int i = 0; i <= Nx + 1; i++)
    {
        for (int j = 0; j <= Ny + 1; j++){
            double tmp = u_exact[I(i, j)] - u[I(i, j)];
            value += tmp * tmp;
        }
    }
    return value;
};

void ComputeOneStep(const std::vector<double>& u, std::vector<double>& new_u, const std::vector<double>& source) {
    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            new_u[I(i, j)] = (1. / A_ii) * ((u[I(i + 1, j)] + u[I(i - 1, j)]) / dx2 + (u[I(i, j + 1)] + u[I(i, j - 1)]) / dy2 - source[I(i, j)]);
        };
    };
};

// ----------------------------------------------------------------------------
// initialization
// void initialize(){
//     for (int i = 0; i <= Nx + 1; i++) {
//     double x = i * dx;
//     for (int j = 0; j <= Ny + 1; j++)
//     {
//         double y = j * dy;
//         if (i == 0)
//         {
//             sol[I(i, j)] = new_sol[I(i, j)] = U_0;
//         }
//         else if (i == Nx + 1){
//             sol[I(i, j)] = new_sol[I(i, j)] = U_0;
//         }
//         else if (j == 0){
//             sol[I(i, j)] = new_sol[I(i, j)] = U_0;
//         }
//         else if (j == Ny + 1){
//             sol[I(i, j)] = new_sol[I(i, j)] = U_0;
//         }
//         else {
//             sol[I(i, j)] = new_sol[I(i, j)] = U_0;
//         }
//         source[I(i, j)] = (double)-(k_x_squared + k_y_squared) * sin(k_x * x) * sin(k_y * y);
//         exact_sol[I(i, j)] = (double) sin(k_x * x) * sin(k_y * y);
//     }
// }
// };

// ----------------------------------------------------------------------------