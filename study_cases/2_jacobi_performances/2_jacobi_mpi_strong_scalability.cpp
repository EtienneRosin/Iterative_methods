// #include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>

#include "../../solvers/jacobi_mpi.hpp"
#include "./2_strong_scalability_parameters.cpp"

void run_jacobi_mpi_test(int max_iterations, double tolerance) {
    int rank, num_processes;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    JacobiMPI jac_mpi(
        Nx, Ny, x_min, x_max, y_min, y_max, 
        dirichlet_condition, dirichlet_condition, 
        dirichlet_condition, dirichlet_condition, 
        laplacien_sin_sin, constant, sin_sin, false);

    jac_mpi.Solve(max_iterations, tolerance);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end - start).count();

    if (rank == 0) {
        std::ofstream results_file;
        results_file.open("../study_cases/2_jacobi_performances/results/jacobi_strong_scalability_results.csv", std::ios::app);
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