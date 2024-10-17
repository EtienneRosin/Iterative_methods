// g++ -std=c++17 jacobi_sequential.cpp -o jacobi_sequential

#include "gauss_seidel_sequential_red_black.hpp"

GaussSeidelSequentialRedBlack::GaussSeidelSequentialRedBlack(
        int Nx, int Ny, double x_min, double x_max, double y_min, double y_max,
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
        OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
        TwoVarsFuncPtr source_func, TwoVarsFuncPtr init_guess_func,
        TwoVarsFuncPtr exact_sol_func, bool verbose) 
        : Nx_(Nx), Ny_(Ny), x_min_(x_min), x_max_(x_max), 
        y_min_(y_min), y_max_(y_max), num_points_((Nx + 2) * (Ny + 2)), 
        dx_((x_max - x_min) / (Nx + 1)), dy_((y_max - y_min) / (Ny + 1)), dx2_(dx_ * dx_), dy2_(dy_ * dy_), 
        sol_(num_points_, 0.0), new_sol_(num_points_, 0.0), source_(num_points_, 0.0), last_l_(-1), class_name_(__func__) {
    
    // if an exact solution function has been provided
    if (exact_sol_func != nullptr){
        exact_sol_.resize(num_points_, 0.0);
    };

    InitializeProblem(left_condition_func, right_condition_func, top_condition_func, bottom_condition_func, source_func, init_guess_func, exact_sol_func);
    // r_init_ = std::sqrt(SquaredResidual(sol_, source_));
    r_init_ = std::sqrt(InitialSquaredResidual(source_));

    if (verbose){
        SaveOnDomain(sol_, "initial_guess");
        SaveOnDomain(source_, "source_term");
        std::cout << "Problem initialized with following properties : \n";
        std::cout << "| Boundaries : [" << x_min_ << ", " << x_max_ << "]x[" << y_min_ << ", " << y_max_ << "]\n";
        std::cout << "| Grid size : " << Nx_ + 2 << "x" << Ny_ + 2 << "\n";
        std::cout << "| dx : " << dx_ << ", dy : " << dy_ << "\n";
        std::cout << "| r^(0) : " << r_init_ << "\n";
        if (exact_sol_func != nullptr){
            std::cout << "| e^(0) : " << std::sqrt(SquaredError(sol_, exact_sol_)) << "\n";
            SaveOnDomain(exact_sol_, "exact_solution");
        };
    }
};


void GaussSeidelSequentialRedBlack::InitializeProblem(
    OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
    OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
    TwoVarsFuncPtr source_func, TwoVarsFuncPtr init_guess_func,
    TwoVarsFuncPtr exact_sol_func) {
    for (int i = 0; i <= Nx_ + 1; i++) {
        double x = x_min_ + i * dx_;
        for (int j = 0; j <= Ny_ + 1; j++)
        {
            double y = y_min_ + j * dy_;
            if (i == 0)
            {
                sol_[I(i, j)] = new_sol_[I(i, j)] = left_condition_func(x);
            }
            else if (i == Nx_ + 1){
                sol_[I(i, j)] = new_sol_[I(i, j)] = right_condition_func(x);
            }
            else if (j == 0){
                sol_[I(i, j)] = new_sol_[I(i, j)] = bottom_condition_func(y);
            }
            else if (j == Ny_ + 1){
                sol_[I(i, j)] = new_sol_[I(i, j)] = top_condition_func(y);
            }
            else {
                sol_[I(i, j)] = init_guess_func(x, y);
            }
            source_[I(i, j)] = source_func(x, y);
            if (exact_sol_func != nullptr){
                exact_sol_[I(i, j)] = exact_sol_func(x, y);
            }
        }
    }
};


int GaussSeidelSequentialRedBlack::I(int i, int j)
{
    return i + j * (Nx_ + 2);
}

bool GaussSeidelSequentialRedBlack::IsRed(int i, int j){
    return (i + j) % 2 == 0;
}

double GaussSeidelSequentialRedBlack::Ax(const std::vector<double> &u, int i, int j){
    return (u[I(i + 1, j)] - 2. * u[I(i, j)] + u[I(i - 1, j)]) / dx2_ + (u[I(i, j + 1)] - 2. * u[I(i, j)] + u[I(i, j - 1)]) / dy2_;
}

double GaussSeidelSequentialRedBlack::Nx(const std::vector<double> &u, const std::vector<double> &new_u, int i, int j) {
    return -(u[I(i + 1, j)] + new_u[I(i - 1, j)]) / dx2_ - (u[I(i, j + 1)] + new_u[I(i, j - 1)]) / dy2_;
}

double GaussSeidelSequentialRedBlack::InitialSquaredResidual(const std::vector<double> &source)
{
    double value = 0.0;
    for (int i = 1; i <= Nx_; i++)
    {
        for (int j = 1; j <= Ny_; j++){
            double tmp = source[I(i, j)];
            value += tmp * tmp;
        }
    }
    return value /((Nx_ + 2)*(Ny_ + 2));
};

double GaussSeidelSequentialRedBlack::SquaredResidual(const std::vector<double> &u, const std::vector<double> &source){
    double value = 0.0;
    for (int i = 1; i <= Nx_; i++)
    {
        for (int j = 1; j <= Ny_; j++){
            double tmp = source[I(i, j)] - Ax(u, i, j);
            value += tmp * tmp;
        }
    }
    return value/((Nx_ + 2)*(Ny_ + 2));
}

double GaussSeidelSequentialRedBlack::SquaredError(const std::vector<double> &u, const std::vector<double> &u_exact){
    double value = 0.0;
    for (int i = 0; i <= Nx_ + 1; i++)
    {
        for (int j = 0; j <= Ny_ + 1; j++){
            double tmp = u_exact[I(i, j)] - u[I(i, j)];
            value += tmp * tmp;
        }
    }
    return value/((Nx_ + 2)*(Ny_ + 2));
}

void GaussSeidelSequentialRedBlack::ComputeOneStep(const std::vector<double> &u, std::vector<double> &new_u, const std::vector<double> &source){
    double A_ii_minus_one = - 0.5 * (dx2_ * dy2_) / (dx2_ + dy2_);
    for (int i = 1; i <= Nx_; i++)
    {
        for (int j = 1; j <= Ny_; j++) {
            if (IsRed(i,j)) {
                new_u[I(i, j)] = A_ii_minus_one * (source[I(i, j)] + Nx(u, u, i, j));
            }
        };
    };
    for (int i = 1; i <= Nx_; i++)
    {
        for (int j = 1; j <= Ny_; j++) {
            if (not IsRed(i,j)) {
                new_u[I(i, j)] = A_ii_minus_one * (source[I(i, j)] + Nx(new_u, new_u, i, j));
            }
        };
    };
}

void GaussSeidelSequentialRedBlack::Solve(int max_iterations, double epsilon, bool verbose) {
    for (int l = 1; l <= max_iterations; l++) {
        ComputeOneStep(sol_, new_sol_, source_);
        sol_.swap(new_sol_);

        r_ = std::sqrt(SquaredResidual(sol_, source_));
        if (verbose){
            std::cout << "l : " << l << ", last relative residual : " << r_ / r_init_;
            if (!exact_sol_.empty()) {
                error_ = std::sqrt(SquaredError(sol_, exact_sol_));
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
        error_ = std::sqrt(SquaredError(sol_, exact_sol_));
        std::cout << "Last error :  " << error_ << "\n";
    }

    SaveOnDomain(sol_, "last_iteration_solution");
}

void GaussSeidelSequentialRedBlack::SaveOnDomain(const std::vector<double> &field, std::string file_name, std::string folder, std::string format){
    std::string full_name = folder + class_name_ + "_" + file_name + format;
    std::ofstream file;
    file.open(full_name);
    file << "x; y; u\n";
    for (int i = 0; i <= Nx_ + 1; i++) {
        for (int j = 0; j <= Ny_ + 1; j++) {
            file << x_min_ + i * dx_ << "; " << y_min_ + j * dy_ << "; " << field[I(i, j)] << "\n";
        }
    }
    file.close();
}