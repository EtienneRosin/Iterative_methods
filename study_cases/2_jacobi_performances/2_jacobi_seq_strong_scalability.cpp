// #include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>

#include "../../solvers/jacobi_sequential.hpp"
#include "./2_strong_scalability_parameters.cpp"

void run_jacobi_seq_test(int max_iterations, double tolerance) {
    auto start = std::chrono::high_resolution_clock::now();
    JacobiSequential jac_seq(
        Nx, Ny, x_min, x_max, y_min, y_max, 
        dirichlet_condition, dirichlet_condition, 
        dirichlet_condition, dirichlet_condition, 
        laplacien_sin_sin, constant, sin_sin, false);
    


    jac_seq.Solve(max_iterations, tolerance);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end - start).count();

    std::ofstream results_file;
        results_file.open("../study_cases/2_jacobi_performances/results/jacobi_strong_scalability_results.csv", std::ios::app);
        results_file << -1 << "; " << elapsed_time << "\n";
        results_file.close();
}

int main() {

    int max_iterations = 200000;
    double tolerance = 1e-6; 
    run_jacobi_seq_test(max_iterations, tolerance);
    return 0;
}