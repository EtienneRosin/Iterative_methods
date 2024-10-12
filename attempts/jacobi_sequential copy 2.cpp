#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <chrono>

// g++ -std=c++17 jacobi_sequential.cpp -o jacobi_sequential


// Definition of functions pointers 
typedef double (*OneVarFuncPtr)(double);            // of one variable use for boundary condition
typedef double (*TwoVarsFuncPrt)(double, double);   // of 2 variables for domain interior


class JacobiSequential {
public:
    // domain properties
    const int Nx_;  //! number of points within the domain in the x-direction
    const int Ny_;  //! number of points within the domain in the y-direction
    const int num_points_; //! total number of points of the domain (include the boundary)

    const double x_min_;    //! domain x-direction lower bound
    const double x_max_;    //! domain x-direction upper bound
    const double y_min_;    //! domain y-direction lower bound
    const double y_max_;    //! domain y-direction upper bound

    const double dx_;   //! step size in the x-direction
    const double dy_;   //! step size in the y-direction

    // problem properties
    std::vector<double> sol_;       //! current iteration solution (e.g. x^(l))
    std::vector<double> new_sol_;   //! next iteration solution (e.g. x^(l+1))
    std::vector<double> source_;    //! source term of the PDE
    std::vector<double> exact_sol_; //! exact solution (defined only if the exact solution has been provided)

    // solver properties
    double r_init_; //! initial residual (e.g. ||r^(0)||)
    double r_; //! current residual (e.g. ||r^(l)||)
    double error_; //! current error (defined only if the exact solution has been provided)
    int last_l_; //! last iteration (-1 if no convergence)
    std::string class_name_; //! name of the class (used for saving)

    // ----------------------------------------------------
    JacobiSequential(
        int Nx, int Ny, double x_min, double x_max, double y_min, double y_max,
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
        OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
        TwoVarsFuncPrt source_func, TwoVarsFuncPrt initi_guess_func,
        TwoVarsFuncPrt exact_sol_func = nullptr, bool verbose = false);

    void InitializeProblem(
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
    OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
    TwoVarsFuncPrt source_func, TwoVarsFuncPrt init_guess_func,
    TwoVarsFuncPrt exact_sol_func);

    int I(int i, int j);

    float ComputeSquaredResidual(const std::vector<double> &u, const std::vector<double> &source);
    // ComputeSquaredResidual();

    float ComputeSquaredError(const std::vector<double> &u, const std::vector<double> &u_exact);
    // ComputeSquaredError();

    void ComputeOneStep(const std::vector<double> &u, std::vector<double> &new_u, const std::vector<double> &source);

    // ComputeOneStep();

    void Solve(int max_iteration, double epsilon, bool verbose = false);

    void SaveOnDomain(const std::vector<double> &field, std::string file_name, std::string format = ".csv");
};

JacobiSequential::JacobiSequential(
    int Nx, int Ny, double x_min, double x_max, double y_min, double y_max,
    OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
    OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
    TwoVarsFuncPrt source_func, TwoVarsFuncPrt init_guess_func,
    TwoVarsFuncPrt exact_sol_func, bool verbose) : Nx_(Nx), Ny_(Ny), x_min_(x_min), x_max_(x_max), y_min_(y_min), y_max_(y_max), num_points_((Nx + 2) * (Ny + 2)), dx_((x_max - x_min) / (Nx + 1)), dy_((y_max - y_min) / (Ny + 1)), sol_(num_points_, 0.0), new_sol_(num_points_, 0.0), source_(num_points_, 0.0), last_l_(-1), class_name_(__func__) {
    
    // if an exact solution function has been provided
    if (exact_sol_func != nullptr){
        exact_sol_.resize(num_points_, 0.0);
    };

    InitializeProblem(left_condition_func, right_condition_func, top_condition_func, bottom_condition_func, source_func, init_guess_func, exact_sol_func);
    r_init_ = std::sqrt(ComputeSquaredResidual(sol_, source_));
    if (verbose){
        std::cout << "Problem initialized with following properties : \n";
        std::cout << "| Boundaries : [" << x_min_ << ", " << x_max_ << "]x[" << y_min_ << ", " << y_max_ << "]\n";
        std::cout << "| Grid size : " << Nx_ + 2 << "x" << Ny_ + 2 << "\n";
        std::cout << "| dx : " << dx_ << ", dy : " << dy_ << "\n";
        std::cout << "| r^(0) : " << r_init_ << "\n";
        if (exact_sol_func != nullptr){
            std::cout << "| e^(0) : " << std::sqrt(ComputeSquaredError(sol_, exact_sol_)) << "\n";
        };
    }
};

int JacobiSequential::I(int i, int j){
    return i + j * (Nx_ + 2);
}

void JacobiSequential::InitializeProblem(
    OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
    OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
    TwoVarsFuncPrt source_func, TwoVarsFuncPrt init_guess_func,
    TwoVarsFuncPrt exact_sol_func) {
    for (int i = 0; i <= Nx_ + 1; i++) {
        double x = x_min_ + i * dx_;
        for (int j = 0; j <= Ny_ + 1; j++)
        {
            double y = y_min_ + j * dy_;
            if (i == 0)
            {
                // sol_[I(i, j)] = new_sol_[I(i, j)] = U_0;
                sol_[I(i, j)] = new_sol_[I(i, j)] = left_condition_func(x);
                // sol[I(i, j)] = new_sol[I(i, j)] = (double) sin(k_x * x) * sin(k_y * y);
            }
            else if (i == Nx_ + 1){
                // sol_[I(i, j)] = new_sol_[I(i, j)] = U_0;
                sol_[I(i, j)] = new_sol_[I(i, j)] = right_condition_func(x);
                // sol[I(i, j)] = new_sol[I(i, j)] = (double) sin(k_x * x) * sin(k_y * y);
            }
            else if (j == 0){
                // sol_[I(i, j)] = new_sol_[I(i, j)] = U_0;
                sol_[I(i, j)] = new_sol_[I(i, j)] = bottom_condition_func(y);
                // sol[I(i, j)] = new_sol[I(i, j)] = (double) sin(k_x * x) * sin(k_y * y);
            }
            else if (j == Ny_ + 1){
                // sol_[I(i, j)] = new_sol[_I(i, j)] = U_0;
                sol_[I(i, j)] = new_sol_[I(i, j)] = top_condition_func(y);
                // sol[I(i, j)] = new_sol[I(i, j)] = (double) sin(k_x * x) * sin(k_y * y);
            }
            else {
                sol_[I(i, j)] = new_sol_[I(i, j)] = init_guess_func(x, y);
                // sol[I(i, j)] = new_sol[I(i, j)] = (double) sin(k_x * x) * sin(k_y * y);
            }
            // source_[I(i, j)] = (double)-(k_x_squared + k_y_squared) * sin(k_x * x) * sin(k_y * y);
            // exact_sol_[I(i, j)] = (double) sin(k_x * x) * sin(k_y * y);
            source_[I(i, j)] = source_func(x, y);
            if (exact_sol_func != nullptr){
                exact_sol_[I(i, j)] = exact_sol_func(x, y);
            }
        }
    }
};


float JacobiSequential::ComputeSquaredResidual(const std::vector<double> &u, const std::vector<double> &source){
    double value = 0.0;
    double dx2 = dx_ * dx_;
    double dy2 = dy_ * dy_;
    for (int i = 1; i <= Nx_; i++)
    {
        for (int j = 1; j <= Ny_; j++){
            double tmp = source[I(i, j)] - ((u[I(i + 1, j)] - 2. * u[I(i, j)] + u[I(i - 1, j)]) / dx2 + (u[I(i, j + 1)] - 2. * u[I(i, j)] + u[I(i, j - 1)]) / dy2);
            value += tmp * tmp;
        }
    }
    return value;
};


float JacobiSequential::ComputeSquaredError(const std::vector<double> &u, const std::vector<double> &u_exact){
    double value = 0.0;
    // double tmp = 0.0;
    for (int i = 0; i <= Nx_ + 1; i++)
    {
        for (int j = 0; j <= Ny_ + 1; j++){
            double tmp = u_exact[I(i, j)] - u[I(i, j)];
            value += tmp * tmp;
        }
    }
    return value;
};

void JacobiSequential::ComputeOneStep(const std::vector<double>& u, std::vector<double>& new_u, const std::vector<double>& source) {
    double dx2 = dx_ * dx_;
    double dy2 = dy_ * dy_;
    double A_ii = -2. * (1. / dx2 + 1. / dy2);
    double A_ii_minus_one = 0.5 * (dx2 * dy2) / (dx2 + dy2);
    std::cout << "A_ii_minus_one : " << A_ii_minus_one << "\n";
    for (int i = 1; i <= Nx_; i++)
    {
        for (int j = 1; j <= Ny_; j++) {
            // new_u[I(i, j)] = (1. / A_ii) * (- (u[I(i + 1, j)] + u[I(i - 1, j)]) / dx2 - (u[I(i, j + 1)] + u[I(i, j - 1)]) / dy2 + source[I(i, j)]);
            new_u[I(i, j)] = A_ii_minus_one * ( (u[I(i + 1, j)] + u[I(i - 1, j)]) / dx2 + (u[I(i, j + 1)] + u[I(i, j - 1)]) / dy2) - A_ii_minus_one * source[I(i, j)];
        };
    };
};


void JacobiSequential::Solve(int max_iterations, double epsilon, bool verbose) {
    for (int l = 1; l <= max_iterations; l++) {
        ComputeOneStep(sol_, new_sol_, source_);
        sol_.swap(new_sol_);

        r_ = std::sqrt(ComputeSquaredResidual(sol_, source_));
        if (verbose){
            std::cout << "l : " << l << ", last residual : " << r_;
            if (!exact_sol_.empty()) {
                error_ = std::sqrt(ComputeSquaredError(sol_, exact_sol_));
                std::cout << ", error :  " << error_;
            }
            std::cout << "\n";
        }
        if (r_ <= epsilon * r_init_){

            std::cout << "Converged after " << l << " iterations, last residual : " << r_ << "\n";
            last_l_ = l;
            break;
        }
    }
    if (last_l_ == -1){
        std::cout << "No convergence, last residual : " << r_ << "\n";
    }

    if (!exact_sol_.empty()) {
        error_ = std::sqrt(ComputeSquaredError(sol_, exact_sol_));
        std::cout << "Last error :  " << error_ << "\n";
    }

    SaveOnDomain(sol_, "last_iteration_solution");
}

void JacobiSequential::SaveOnDomain(const std::vector<double> &field, std::string file_name, std::string format) {
    std::string full_name = "./results" + class_name_ + "_" + file_name + format;
    std::ofstream file;
    file.open(full_name);
    file << "x; y; u\n";
    for (int i = 0; i <= Nx_ + 1; i++) {
        for (int j = 0; j <= Ny_ + 1; j++) {
            file << x_min_ + i * dx_ << "; " << y_min_ + j * dy_ << "; " << field[I(i, j)] << "\n";
        }
    }
    file.close();
};


// ----------------------------------------------------------------------------
double a = 2 * M_PI;
double b = 2 * M_PI;

double k_x = 2. * M_PI / a;
double k_y = 2. * M_PI / b;

double k_x_squared = k_x*k_x;
double k_y_squared = k_y*k_y;

double U_0 = 0.;

double dirichlet_condition(double m)
{
    return U_0;
};

double constant(double x, double y){
    return U_0;
}

double sin_sin(double x, double y) {
    return (double) sin(k_x * x) * sin(k_y * y);
};

double laplacien_sin_sin(double x, double y) {
    return (double)-(k_x_squared + k_y_squared) * sin(k_x * x) * sin(k_y * y);
};




// main to run one time with terminal -----------------------------------------
// int main() {
//     int n = 100;
//     double x_min = 0.;
//     double y_min = 0.;
//     double x_max = a;
//     double y_max = b;
//     // JacobiSequential jac(n, n )

//     JacobiSequential jac_seq(
//         n, n, x_min, x_max, y_min, y_max, 
//         dirichlet_condition, dirichlet_condition, dirichlet_condition, dirichlet_condition, 
//         laplacien_sin_sin, constant, sin_sin, true);

//     jac_seq.Solve(20000, 1e-6);
//     return 0;
// };
int main() {
    int n = 100;
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
        laplacien_sin_sin, sin_sin, sin_sin, true);
    auto end_init = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed_init = end_init - start_init;
    std::cout << "Temps d'initialisation : " << elapsed_init.count() << " secondes\n";

    // Mesurer le temps de la méthode Solve
    auto start_solve = std::chrono::high_resolution_clock::now();
    jac_seq.Solve(20000, 1e-6, false);
    auto end_solve = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed_solve = end_solve - start_solve;
    std::cout << "Execution time : " << elapsed_solve.count() << " secondes\n";

    // Calculer le temps total
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_total = end_total - start_total;
    std::cout << "Total time : " << elapsed_total.count() << " secondes\n";

    return 0;
}

// main for assymptotic accuracy ----------------------------------------------
// int main() {
//     if (!std::filesystem::exists("results")) {
//         std::filesystem::create_directory("results");
//     }
//     // int n = 100;
//     double x_min = 0.;
//     double y_min = 0.;
//     double x_max = a;
//     double y_max = b;
//     // JacobiSequential jac(n, n )


//     // --------------------------------
//     int n_min = 50; 
//     int n_max = 400; 
//     // double epsilon_min = 1e-1; 
//     double epsilon_min = 1e-6;
//     double epsilon_max = 1e-6;
//     int maxIterations = 200000;
//     // --------------------------------
//     std::filesystem::path full_name = "./jacobi_sequential_asymptotic_accuracy.csv";
//     std::ofstream log_file(full_name);

//     if (!log_file.is_open()) {
//         std::cerr << "Error while opening the log file" << std::endl;
//         return 1;
//     }

//     log_file << "epsilon; h; error; last_iteration\n";

//     for (double epsilon = epsilon_min; epsilon >= epsilon_max; epsilon /= 10) {
//         for (int n = n_min; n <= n_max; n += 50) {
//             double h = x_max / (n + 1);
//             std::cout << "n : " << n << ", epsilon : " << epsilon << "\n";
//             JacobiSequential jac_seq(
//                 n, n, x_min, x_max, y_min, y_max,
//                 dirichlet_condition, dirichlet_condition,
//                 dirichlet_condition, dirichlet_condition,
//                 laplacien_sin_sin, constant, sin_sin, false);

//             jac_seq.Solve(maxIterations, epsilon_max);

//             log_file << epsilon << "; " << h << "; " << jac_seq.error_ << "; " << jac_seq.last_l_ << "\n";
//         }
//     }
//     log_file.close();
//     return 0;
// };
