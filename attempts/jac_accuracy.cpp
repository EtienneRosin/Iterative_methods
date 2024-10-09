#include <iostream>
#include <vector>
#include <cmath>

double a = 2.;
double b = 2.;
double U_0 = 0.;


int I(int i, int j, int Nx) {
    return i + (Nx + 2) * j;
}

float ComputeSquaredResidual(const std::vector<double>& u, const std::vector<double>& source, int Nx, int Ny, double dx2, double dy2) {
    double value = 0.0;
    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            double tmp = source[I(i, j, Nx)] - ((u[I(i + 1, j, Nx)] - 2.*u[I(i, j, Nx)] + u[I(i - 1, j, Nx)]) / dx2 
                                               + (u[I(i, j + 1, Nx)] - 2.*u[I(i, j, Nx)] + u[I(i, j - 1, Nx)]) / dy2);
            value += tmp * tmp;
        }
    }
    return value;
}

float ComputeSquaredError(const std::vector<double>& u, const std::vector<double>& u_exact, int Nx, int Ny) {
    double value = 0.0;
    for (int i = 0; i <= Nx + 1; i++) {
        for (int j = 0; j <= Ny + 1; j++) {
            double tmp = u_exact[I(i, j, Nx)] - u[I(i, j, Nx)];
            value += tmp * tmp;
        }
    }
    return value;
}

void ComputeOneStep(const std::vector<double>& u, std::vector<double>& new_u, const std::vector<double>& source, int Nx, int Ny, double dx2, double dy2, double A_ii) {
    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            new_u[I(i, j, Nx)] = (1. / A_ii) * (-(u[I(i + 1, j, Nx)] + u[I(i - 1, j, Nx)]) / dx2 
                                               - (u[I(i, j + 1, Nx)] + u[I(i, j - 1, Nx)]) / dy2 
                                               + source[I(i, j, Nx)]);
        }
    }
}

// Initialisation de la grille
void initialize(int Nx, int Ny, double dx, double dy, std::vector<double>& sol, std::vector<double>& new_sol, std::vector<double>& exact_sol, std::vector<double>& source) {
    double k_x = 2. * M_PI / a;
    double k_y = 2. * M_PI / b;
    double k_x_squared = k_x * k_x;
    double k_y_squared = k_y * k_y;

    for (int i = 0; i <= Nx + 1; i++) {
        double x = i * dx;
        for (int j = 0; j <= Ny + 1; j++) {
            double y = j * dy;
            if (i == 0 || i == Nx + 1 || j == 0 || j == Ny + 1) {
                sol[I(i, j, Nx)] = new_sol[I(i, j, Nx)] = U_0;
            } else {
                sol[I(i, j, Nx)] = new_sol[I(i, j, Nx)] = U_0;
            }
            source[I(i, j, Nx)] = -(k_x_squared + k_y_squared) * sin(k_x * x) * sin(k_y * y);
            exact_sol[I(i, j, Nx)] = sin(k_x * x) * sin(k_y * y);
        }
    }
}

int main() {
    int max_iterations = 50000;
    double epsilon = 1e-6;

    // Boucle sur différentes tailles de grille
    std::vector<int> grid_sizes = {10, 20, 30, 40, 50, 100, 150, 200, 250};
    for (int n : grid_sizes) {
        int Nx = n;
        int Ny = n;
        int num_points = (Nx + 2) * (Ny + 2);
        
        double dx = a / (Nx + 1);
        double dy = b / (Ny + 1);
        double dx2 = dx * dx;
        double dy2 = dy * dy;
        double A_ii = -2. * (1. / dx2 + 1. / dy2);
        
        std::vector<double> sol(num_points, 0.0);
        std::vector<double> new_sol(num_points, 0.0);
        std::vector<double> exact_sol(num_points, 0.0);
        std::vector<double> source(num_points, 0.0);

        initialize(Nx, Ny, dx, dy, sol, new_sol, exact_sol, source);

        double initial_residual = std::sqrt(ComputeSquaredResidual(sol, source, Nx, Ny, dx2, dy2));
        double residual = 0.0;
        int last_iteration = -1;

        for (int l = 1; l <= max_iterations; l++) {
            ComputeOneStep(sol, new_sol, source, Nx, Ny, dx2, dy2, A_ii);
            sol.swap(new_sol);
            residual = std::sqrt(ComputeSquaredResidual(sol, source, Nx, Ny, dx2, dy2));
            if (residual <= epsilon * initial_residual) {
                last_iteration = l;
                break;
            }
        }

        double error = std::sqrt(ComputeSquaredError(sol, exact_sol, Nx, Ny));
        std::cout << "Grid size: " << n << "x" << n << ", Error: " << error << "\n";
    }

    return 0;
}