#include "./3_parameters.hpp"

#include "../../solvers/jacobi_mpi.hpp"
#include "../../solvers/gauss_seidel_mpi.hpp"

#include <iostream>
#include <cmath>
#include <chrono>

// void write_results(int num_processes, double elapsed_time) {
//     std::ofstream results_file;
//     results_file.open("../study_cases/2_jacobi_performances/results/jacobi_weak_scalability_results.csv", std::ios::app);
    
//     if (!results_file.is_open()) {
//         std::cerr << "Error opening results file!" << std::endl;
//         return;
//     }
    
//     results_file << num_processes << "; " << elapsed_time << "\n";
//     results_file.close();
// }

// void run_jacobi_mpi_test(int max_iterations, double tolerance) {
//     int rank, num_processes;
    
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
//     MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

//     MPI_Barrier(MPI_COMM_WORLD);
//     auto start = std::chrono::high_resolution_clock::now();
//     int num_y = num_processes * (Ny + 2) - 2;

//     JacobiMPI jac_mpi(
//         Nx, num_y, x_min, x_max, y_min, y_max,
//         dirichlet_condition, dirichlet_condition,
//         dirichlet_condition, dirichlet_condition,
//         laplacien_sin_sin, constant, sin_sin, false);

//     jac_mpi.Solve(max_iterations, tolerance);

//     MPI_Barrier(MPI_COMM_WORLD);
//     auto end = std::chrono::high_resolution_clock::now();
//     double elapsed_time = std::chrono::duration<double>(end - start).count();

//     if (rank == 0) {
//         write_results(num_processes, elapsed_time);
//     }
// }

// // int main(int argc, char** argv) {
// //     MPI_Init(&argc, &argv);

// //     run_jacobi_mpi_test(max_iterations, tolerance);
// //     MPI_Finalize(); 
// //     return 0;
// // }


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, num_processes;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    MPI_Barrier(MPI_COMM_WORLD);
    int num_y = num_processes * (Ny + 2) - 2;

    auto start = std::chrono::high_resolution_clock::now();
    JacobiMPI jac_mpi_solver(
        Nx, num_y, x_min, x_max, y_min, y_max, 
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
        Nx, num_y, x_min, x_max, y_min, y_max, 
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