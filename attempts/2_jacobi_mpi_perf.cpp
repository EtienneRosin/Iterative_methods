#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>

#include "../../solvers/jacobi_MPI.hpp"


double a = 1;
double b = 1;

double k_x = 2. * M_PI / a;
double k_y = 2. * M_PI / b;

double k_x_squared = k_x * k_x;
double k_y_squared = k_y * k_y;

double U_0 = 0.;

double dirichlet_condition(double m)
{
    return U_0;
};

double constant(double x, double y){
    return U_0;
}

double f(double x, double y){
    return 2. * (-a * x + x * x - b * y + y * y);
}

double sin_sin(double x, double y) {
    return sin(k_x * x) * sin(k_y * y);
};

double u(double x, double y){
    return x * y * (a - x) * (b - y);
}

double laplacien_sin_sin(double x, double y) {
    return -(k_x_squared + k_y_squared) * sin_sin(x, y);
};



void run_jacobi_mpi_test(int max_iterations, double tolerance) {
    int rank, num_processes;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    int n = 200;
    double x_min = 0.;
    double y_min = 0.;
    double x_max = a;
    double y_max = b;

    JacobiMPI jac_mpi(
        n, n, x_min, x_max, y_min, y_max, 
        dirichlet_condition, dirichlet_condition, 
        dirichlet_condition, dirichlet_condition, 
        laplacien_sin_sin, constant, sin_sin, false);

    jac_mpi.Solve(max_iterations, tolerance);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end - start).count();

    if (rank == 0) {
        std::ofstream results_file;
        results_file.open("../study_cases/2_jacobi_performances/results/mpi_performance_results.csv", std::ios::app);
        results_file << num_processes << "; " << elapsed_time << "\n";
        results_file.close();
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int max_iterations = 200000;
    double tolerance = 1e-6; 
    run_jacobi_mpi_test(max_iterations, tolerance);

    MPI_Finalize(); 

    return 0;
}