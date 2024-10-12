#ifndef JACOBI_MPI_HPP
#define JACOBI_MPI_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <mpi.h>

#include "utils.hpp"


class JacobiMPI {
public: 
    // domain properties ------------------------------------------------------
    const int Nx_;       //! number of points within the global domain in the x-direction
    const int Ny_;       //! number of points within the global domain in the y-direction
    int num_local_rows_; //! number of rows of the subdomain (include the ghost cells)

    const double x_min_;    //! global domain x-direction lower bound
    const double x_max_;    //! global domain x-direction upper bound
    const double y_min_;    //! global domain y-direction lower bound
    const double y_max_;    //! global domain y-direction upper bound

    const double dx_;   //! step size in the x-direction
    const double dy_;   //! step size in the y-direction
    const double dx2_;
    const double dy2_;

    // problem properties -----------------------------------------------------
    std::vector<double> local_sol_;       //! current iteration local solution (e.g. x^(l))
    std::vector<double> new_local_sol_;   //! next iteration local solution (e.g. x^(l+1))
    std::vector<double> local_source_;    //! local source term of the PDE
    std::vector<double> local_exact_sol_; //! local exact solution (defined only if the exact solution has been provided)
    std::vector<double> global_sol_;      //! solution on the whole domain

    // solver properties ------------------------------------------------------
    double r_init_;          //! initial residual (e.g. ||r^(0)||)
    double r_;               //! current residual (e.g. ||r^(l)||)
    double error_;           //! current error (defined only if the exact solution has been provided)
    int last_l_;             //! last iteration (-1 if no convergence)
    std::string class_name_; //! name of the class (used for saving)

    // process properties -----------------------------------------------------
    int process_rank_;      //! process rank
    int num_processes_;     //! number of procceses
    int j_start_;           //! initial row index of the subdomain

    // Constructors -----------------------------------------------------------
    JacobiMPI(
        int Nx, int Ny, double x_min, double x_max, double y_min, double y_max,
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
        OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
        TwoVarsFuncPrt source_func, TwoVarsFuncPrt init_guess_func,
        TwoVarsFuncPrt exact_sol_func = nullptr, bool verbose = false);

    // Methods ----------------------------------------------------------------
    void InitializeLocalProblem(
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
        OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
        TwoVarsFuncPrt source_func, TwoVarsFuncPrt init_guess_func,
        TwoVarsFuncPrt exact_sol_func = nullptr);

    int I(int i, int j);

    void CheckMPIError(int errCode, const std::string &errMsg);

    double Ax(const std::vector<double> &u, int i, int j);

    double Nx(const std::vector<double> &u, int i, int j);

    double LocalSquaredResidual(const std::vector<double> &u, const std::vector<double> &source);

    double LocalSquaredError(const std::vector<double> &u, const std::vector<double> &u_exact);

    void ComputeOneLocalStep(const std::vector<double> &u, std::vector<double> &new_u, const std::vector<double> &source);

    void Solve(int max_iterations, double epsilon, bool verbose = false);

    void GatherGlobalSolution();

    void ExchangeBoundaries();

    void SaveOnDomain(const std::vector<double> &field, std::string file_name, std::string folder = "./results/", std::string format = ".csv");
};

#endif // JACOBI_MPI_HPP