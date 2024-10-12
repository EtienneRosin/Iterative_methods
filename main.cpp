#include "solvers/jacobi_sequential.hpp"
#include "solvers/jacobi_mpi.hpp"

// #include <mpi.h>

// g++ -std=c++17 main.cpp -o main
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



int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    // std::cout << "p = " << process_rank << "\n";
    double init_start = MPI_Wtime();

    int n = 200;
    double x_min = 0.;
    double y_min = 0.;
    double x_max = a;
    double y_max = b;

    JacobiMPI jac_mpi(
        n, n, x_min, x_max, y_min, y_max, 
        dirichlet_condition, dirichlet_condition, 
        dirichlet_condition, dirichlet_condition, 
        laplacien_sin_sin, constant, sin_sin, true);

    // double init_end = MPI_Wtime();  // Temps de fin de l'initialisation

    // if (process_rank == 0) {
    //     std::cout << "Initialization time : " << init_end - init_start << " secondes" << std::endl;
    // }

    jac_mpi.Solve(200000, 1e-6);

    MPI_Barrier(MPI_COMM_WORLD);


    // double total_end = MPI_Wtime();

    // if (process_rank == 0) {
    //     std::cout << "Total time : " << total_end - init_start << " secondes" << std::endl;
    // }



    // JacobiSequential jac_seq(
    //     n, n, x_min, x_max, y_min, y_max, 
    //     dirichlet_condition, dirichlet_condition, dirichlet_condition, dirichlet_condition, 
    //     laplacien_sin_sin, constant, sin_sin, true);

    // jac_seq.Solve(200000, 1e-6);
    // return 0;

    MPI_Finalize();
    return 0;

    
};