#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include "../../solvers/jacobi_mpi.hpp"
#include "./2_weak_scalability_parameters.cpp"

void write_results(int num_processes, double elapsed_time) {
    std::ofstream results_file;
    results_file.open("../study_cases/2_jacobi_performances/results/jacobi_weak_scalability_results.csv", std::ios::app);
    
    if (!results_file.is_open()) {
        std::cerr << "Error opening results file!" << std::endl;
        return;
    }
    
    results_file << num_processes << "; " << elapsed_time << "\n";
    results_file.close();
}

void run_jacobi_mpi_test(int max_iterations, double tolerance) {
    int rank, num_processes;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    MPI_Barrier(MPI_COMM_WORLD); // Synchronisation avant le chronométrage
    auto start = std::chrono::high_resolution_clock::now();
    int num_y = num_processes * (Ny + 2) - 2;
    // std::cout << "num_y = " << num_y << "\n";

    // if (rank == 0){
    //     std::cout << "P = " << num_processes << ", (n_x + 2)*(n_y + 2) = " << (Nx + 2) * (Ny + 2) << ", (n_x + 2)*(new_n_y + 2) = " << (num_y + 2) * (Nx + 2) << "\n";
    // }

    JacobiMPI jac_mpi(
        Nx, num_y, x_min, x_max, y_min, y_max,
        dirichlet_condition, dirichlet_condition,
        dirichlet_condition, dirichlet_condition,
        laplacien_sin_sin, constant, sin_sin, false);

    jac_mpi.Solve(max_iterations, tolerance);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end - start).count();

    if (rank == 0) {
        write_results(num_processes, elapsed_time);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int max_iterations = 500000;
    double tolerance = 1e-6; 
    run_jacobi_mpi_test(max_iterations, tolerance);

    MPI_Finalize(); 

    return 0;
}