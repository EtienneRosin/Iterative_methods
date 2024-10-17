#include "./1_parameters.hpp"
#include "../../solvers/jacobi_sequential.hpp"
#include "../../solvers/gauss_seidel_sequential.hpp"

std::string folder = "../study_cases/1_asymptotic_accuracy/";
int main()
{
    std::string file_name_jacobi = folder + "jacobi_sequential_asymptotic_accuracy.csv";
    std::cout << file_name_jacobi << "\n";
    std::ofstream log_file_jacobi(file_name_jacobi);

    std::string file_name_gauss_seidel = folder + "gauss_seidel_sequential_asymptotic_accuracy.csv";
    std::ofstream log_file_gauss_seidel(file_name_gauss_seidel);


    if (!log_file_jacobi.is_open()) {
        std::cerr << "Error while opening the Jacobi log file" << std::endl;
        return 1;
    }
    log_file_jacobi << header;

    if (!log_file_gauss_seidel.is_open()) {
        std::cerr << "Error while opening the Gauss-Seidel log file" << std::endl;
        return 1;
    }
    log_file_gauss_seidel << header;


    for (double epsilon = epsilon_min; epsilon >= epsilon_max; epsilon /= 10)
    {
        for (int n = n_min; n <= n_max; n += 10) {
            std::cout << "n : " << n << ", epsilon : " << epsilon << "\n";

            JacobiSequential solver_jacobi(
                n, n, x_min, x_max, y_min, y_max,
                dirichlet_condition, dirichlet_condition,
                dirichlet_condition, dirichlet_condition,
                laplacien_sin_sin, constant, sin_sin, false);
            solver_jacobi.Solve(maxIterations, epsilon_max, false);
            log_file_jacobi << epsilon << "; " << solver_jacobi.dx_ << "; " 
                            << solver_jacobi.error_ << "; " << solver_jacobi.last_l_ << "\n";

            GaussSeidelSequential solver_gauss_seidel(
                n, n, x_min, x_max, y_min, y_max,
                dirichlet_condition, dirichlet_condition,
                dirichlet_condition, dirichlet_condition,
                laplacien_sin_sin, constant, sin_sin, false);
            solver_gauss_seidel.Solve(maxIterations, epsilon_max, false);
            log_file_gauss_seidel << epsilon << "; " << solver_gauss_seidel.dx_ << "; " 
                            << solver_gauss_seidel.error_ << "; " << solver_gauss_seidel.last_l_ << "\n";

        }
    }
    log_file_jacobi.close();
    log_file_gauss_seidel.close();
    return 0;
};
