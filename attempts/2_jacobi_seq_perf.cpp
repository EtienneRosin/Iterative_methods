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

int main() {
    int n = 200;
    double x_min = 0.;
    double y_min = 0.;
    double x_max = a;
    double y_max = b;

    // Mesurer le temps total de l'exécution
    auto start_total = std::chrono::high_resolution_clock::now();

    // Mesurer le temps d'initialisation
    auto start_init = std::chrono::high_resolution_clock::now();
    JacobiSequential jac_seq(
        n, n, x_min, x_max, y_min, y_max, 
        dirichlet_condition, dirichlet_condition, 
        dirichlet_condition, dirichlet_condition, 
        laplacien_sin_sin, constant, sin_sin, false);
    auto end_init = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed_init = end_init - start_init;
    std::cout << "Temps d'initialisation : " << elapsed_init.count() << " secondes\n";

    // Mesurer le temps de la méthode Solve
    auto start_solve = std::chrono::high_resolution_clock::now();
    jac_seq.Solve(200000, 1e-8, false);
    auto end_solve = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed_solve = end_solve - start_solve;
    std::cout << "Execution time : " << elapsed_solve.count() << " secondes\n";

    // Calculer le temps total
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_total = end_total - start_total;
    std::cout << "Total time : " << elapsed_total.count() << " secondes\n";

    return 0;
}