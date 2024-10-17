#include "./2_parameters.hpp"

#include "../../solvers/jacobi_mpi.hpp"
#include "../../solvers/gauss_seidel_mpi.hpp"

#include <iostream>
#include <cmath>
#include <chrono>


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, num_processes;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    JacobiMPI jac_mpi_solver(
        Nx, Ny, x_min, x_max, y_min, y_max, 
        dirichlet_condition, dirichlet_condition, 
        dirichlet_condition, dirichlet_condition, 
        laplacien_sin_sin, constant, sin_sin, false);

    jac_mpi_solver.Solve(max_iterations, epsilon);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end - start).count();

    if (rank == 0) {
        
        std::ofstream results_file;
        results_file.open(file_name_jacobi, std::ios::app);
        results_file << num_processes << "; " << elapsed_time << "\n";
        results_file.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = std::chrono::high_resolution_clock::now();
    GaussSeidelMPI gauss_seidel_mpi_solver(
        Nx, Ny, x_min, x_max, y_min, y_max, 
        dirichlet_condition, dirichlet_condition, 
        dirichlet_condition, dirichlet_condition, 
        laplacien_sin_sin, constant, sin_sin, false);

    gauss_seidel_mpi_solver.Solve(max_iterations, epsilon);

    end = std::chrono::high_resolution_clock::now();
    
    elapsed_time = std::chrono::duration<double>(end - start).count();

    if (rank == 0) {
        std::ofstream results_file;
        results_file.open(file_name_gauss_seidel, std::ios::app);
        results_file << num_processes << "; " << elapsed_time << "\n";
        results_file.close();
    }

    MPI_Finalize(); 

    return 0;
}