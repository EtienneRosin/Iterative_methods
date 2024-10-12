#include "../../solvers/jacobi_sequential.hpp"

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



// int main() {
//     int n = 200;
//     double x_min = 0.;
//     double y_min = 0.;
//     double x_max = a;
//     double y_max = b;



//     JacobiSequential jac_seq(
//         n, n, x_min, x_max, y_min, y_max, 
//         dirichlet_condition, dirichlet_condition, dirichlet_condition, dirichlet_condition, 
//         laplacien_sin_sin, constant, sin_sin, true);

//     jac_seq.Solve(200000, 1e-6);
//     return 0;
// };


int main() {
    // if (!std::filesystem::exists("results")) {
    //     std::filesystem::create_directory("results");
    // }
    // int n = 100;
    double x_min = 0.;
    double y_min = 0.;
    double x_max = a;
    double y_max = b;
    // int Ny = 50;
    // JacobiSequential jac(n, n )

    // --------------------------------
    int n_min = 50; 
    int n_max = 300; 
    // double epsilon_min = 1e-1; 
    double epsilon_min = 1e-4;
    double epsilon_max = 1e-8;
    int maxIterations = 2000000;
    // --------------------------------
    std::filesystem::path full_name = "../study_cases/1_jacobi_seq_asymptotic_accuracy/results/jacobi_sequential_asymptotic_accuracy.csv";
    std::ofstream log_file(full_name);

    if (!log_file.is_open()) {
        std::cerr << "Error while opening the log file" << std::endl;
        return 1;
    }

    log_file << "epsilon; h^2; error; last_iteration\n";

    for (double epsilon = epsilon_min; epsilon >= epsilon_max; epsilon /= 10) {
        for (int n = n_min; n <= n_max; n += 10) {
            // double h_x = (x_max - x / (n + 1);
            // double h_y = x_max / (n + 1);
            std::cout << "n : " << n << ", epsilon : " << epsilon << "\n";
            JacobiSequential jac_seq(
                n, n, x_min, x_max, y_min, y_max,
                dirichlet_condition, dirichlet_condition,
                dirichlet_condition, dirichlet_condition,
                laplacien_sin_sin, constant, sin_sin, false);

            jac_seq.Solve(maxIterations, epsilon_max);

            log_file << epsilon << "; " << jac_seq.dx_ * jac_seq.dy_ << "; " << jac_seq.error_ << "; " << jac_seq.last_l_ << "\n";
        }
    }
    log_file.close();
    return 0;
};
