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
    using OneVarFuncPtr = utils::OneVarFuncPtr;
    using TwoVarsFuncPtr = utils::TwoVarsFuncPtr;
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
    /*!
      \brief Constructor to initialize the JacobiSequential solver with the grid parameters, boundary conditions, and source term.

      \param Nx Number of grid points along the x-axis.
      \param Ny Number of grid points along the y-axis.
      \param x_min Minimum value of the x-axis.
      \param x_max Maximum value of the x-axis.
      \param y_min Minimum value of the y-axis.
      \param y_max Maximum value of the y-axis.
      \param left_condition_func Function that defines the left boundary condition.
      \param right_condition_func Function that defines the right boundary condition.
      \param top_condition_func Function that defines the top boundary condition.
      \param bottom_condition_func Function that defines the bottom boundary condition.
      \param source_func Function that defines the source term.
      \param init_guess_func Function that provides the initial guess for the solution.
      \param exact_sol_func (Optional) Function that provides the exact solution for error calculation.
      \param verbose (Optional) If true, display some information about the initialization
    */
    JacobiMPI(
        int Nx, int Ny, double x_min, double x_max, double y_min, double y_max,
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
        OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
        TwoVarsFuncPtr source_func, TwoVarsFuncPtr init_guess_func,
        TwoVarsFuncPtr exact_sol_func = nullptr, bool verbose = false);

    // Methods ----------------------------------------------------------------
    /*!
      \brief Initialize the local fields

      \param left_condition_func Function that defines the left boundary condition.
      \param right_condition_func Function that defines the right boundary condition.
      \param top_condition_func Function that defines the top boundary condition.
      \param bottom_condition_func Function that defines the bottom boundary condition.
      \param source_func Function that defines the source term.
      \param init_guess_func Function that provides the initial guess for the solution.
      \param exact_sol_func (Optional) Function that provides the exact solution for error calculation.
    */
    void InitializeLocalProblem(
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
        OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
        TwoVarsFuncPtr source_func, TwoVarsFuncPtr init_guess_func,
        TwoVarsFuncPtr exact_sol_func = nullptr);

    /*!
      \brief Get the linear index associated to (i,j)

      \param i x-component index
      \param j y-component index
      \return I = i + j * (Nx + 2) : the linear index
    */
    int I(int i, int j);

    /*!
      \brief Check for MPI errors and handle them accordingly.

      This function checks if an MPI operation resulted in an error. 
      If an error occurs, it logs the error message to std::cerr and 
      terminates the MPI environment using MPI_Abort, which stops all 
      running processes.

      \param errCode The MPI error code returned by the MPI function.
      \param errMsg A message describing the operation that failed.
    */
    void CheckMPIError(int errCode, const std::string &errMsg);

    /*!
      \brief Compute the coefficient (Ax)_I

      \param i x-component index
      \param j y-component index
      \return (Ax)_I
    */
    double Ax(const std::vector<double> &u, int i, int j);


    /*!
      \brief Compute the coefficient (Nx)_I

      \param i x-component index
      \param j y-component index
      \return (Nx)_I
    */
    double Nx(const std::vector<double> &u, int i, int j);

    /*!
      \brief Compute the squared residual on the process's subdomain

      \param u local solution vector
      \param source local source vector
      \return local squared residual
    */
    double LocalSquaredResidual(const std::vector<double> &u, const std::vector<double> &source);

    /*!
      \brief Compute the initial squared residual on the process's subdomain

      \param source source vector
      \return initial local squared residual
    */
    double LocalInitialSquaredResidual(const std::vector<double> &source);

    /*!
      \brief Compute the squared error on the process's subdomain

      \param u local solution vector
      \param u_exact local exact solution vector
      \return local squared error
    */
    double LocalSquaredError(const std::vector<double> &u, const std::vector<double> &u_exact);

    /*!
      \brief Compute one step of the Jacobi algorithm within the process's subdomain

      \param u local solution vector
      \param new_u new local solution vector
      \param source local source vector
    */
    void ComputeOneLocalStep(const std::vector<double> &u, std::vector<double> &new_u, const std::vector<double> &source);

    /*!
      \brief Solve the linear system

      \param max_iterations maximum number of iterations of the Jacobi algorithm
      \param epsilon tolerance for the stopping criteria
      \param verbose (Optional) if true display the residual (and potentatially the error if the exact solution has been provided in the constructor) at each iteration
    */
    void Solve(int max_iterations, double epsilon, bool verbose = false);

    /*!
      \brief Gather all local solutions to the root process
    */
    void GatherGlobalSolution();

    /*!
      \brief Exchange boundary values with the process's neighbours
    */
    void ExchangeBoundaries();

    /*!
      \brief Save the given GLOBAL field defined on the whole domain in a file

      \param field considered field (here it is used to save the global solution)
      \param file_name file name 
      \param folder (Optionnal) folder where to save the file (default: ../results/)
      \param format (Optionnal) format of the file (default: .csv)
    */
    void SaveOnDomain(const std::vector<double> &field, std::string file_name, std::string folder = "../results/", std::string format = ".csv");
};

#endif // JACOBI_MPI_HPP