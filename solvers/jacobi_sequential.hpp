#ifndef JACOBI_SEQUENTIAL_HPP
#define JACOBI_SEQUENTIAL_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <chrono>


#include "utils.hpp"


class JacobiSequential {
public: 
    // domain properties ------------------------------------------------------
    const int Nx_;  //! number of points within the domain in the x-direction
    const int Ny_;  //! number of points within the domain in the y-direction
    const int num_points_; //! total number of points of the domain (include the boundary)

    const double x_min_;    //! domain x-direction lower bound
    const double x_max_;    //! domain x-direction upper bound
    const double y_min_;    //! domain y-direction lower bound
    const double y_max_;    //! domain y-direction upper bound

    const double dx_;   //! step size in the x-direction
    const double dy_;   //! step size in the y-direction
    const double dx2_;
    const double dy2_;

    // problem properties -----------------------------------------------------
    std::vector<double> sol_;       //! current iteration solution (e.g. x^(l))
    std::vector<double> new_sol_;   //! next iteration solution (e.g. x^(l+1))
    std::vector<double> source_;    //! source term of the PDE
    std::vector<double> exact_sol_; //! exact solution (defined only if the exact solution has been provided)

    // solver properties ------------------------------------------------------
    double r_init_;          //! initial residual (e.g. ||r^(0)||)
    double r_;               //! current residual (e.g. ||r^(l)||)
    double error_;           //! current error (defined only if the exact solution has been provided)
    int last_l_;             //! last iteration (-1 if no convergence)
    std::string class_name_; //! name of the class (used for saving)

    // Constructors -----------------------------------------------------------
    JacobiSequential(
        int Nx, int Ny, double x_min, double x_max, double y_min, double y_max,
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
        OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
        TwoVarsFuncPrt source_func, TwoVarsFuncPrt init_guess_func,
        TwoVarsFuncPrt exact_sol_func = nullptr, bool verbose = false);

    // Methods ----------------------------------------------------------------
    void InitializeProblem(
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
        OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
        TwoVarsFuncPrt source_func, TwoVarsFuncPrt init_guess_func,
        TwoVarsFuncPrt exact_sol_func = nullptr);

    int I(int i, int j);

    double Ax(const std::vector<double> &u, int i, int j);

    double Nx(const std::vector<double> &u, int i, int j);

    double SquaredResidual(const std::vector<double> &u, const std::vector<double> &source);

    double SquaredError(const std::vector<double> &u, const std::vector<double> &u_exact);

    void ComputeOneStep(const std::vector<double> &u, std::vector<double> &new_u, const std::vector<double> &source);

    void Solve(int max_iterations, double epsilon, bool verbose = false);

    void SaveOnDomain(const std::vector<double> &field, std::string file_name, std::string folder = "./results/", std::string format = ".csv");
};

#endif // JACOBI_SEQUENTIAL_HPP