#ifndef GAUSS_SEIDEL_SEQUENTIAL_HPP
#define GAUSS_SEIDEL_SEQUENTIAL_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
// #include <filesystem>
#include <chrono>


#include "utils.hpp"


class GaussSeidelSequential {
public: 
    using OneVarFuncPtr = utils::OneVarFuncPtr;
    using TwoVarsFuncPtr = utils::TwoVarsFuncPtr;
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
    GaussSeidelSequential(
        int Nx, int Ny, double x_min, double x_max, double y_min, double y_max,
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
        OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
        TwoVarsFuncPtr source_func, TwoVarsFuncPtr init_guess_func,
        TwoVarsFuncPtr exact_sol_func = nullptr, bool verbose = false);

    // Methods ----------------------------------------------------------------
    /*!
      \brief Initialize the fields

      \param left_condition_func Function that defines the left boundary condition.
      \param right_condition_func Function that defines the right boundary condition.
      \param top_condition_func Function that defines the top boundary condition.
      \param bottom_condition_func Function that defines the bottom boundary condition.
      \param source_func Function that defines the source term.
      \param init_guess_func Function that provides the initial guess for the solution.
      \param exact_sol_func (Optional) Function that provides the exact solution for error calculation.
    */
    void InitializeProblem(
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
      \brief Compute the coefficient (Ax)_I

      \param i x-component index
      \param j y-component index
      \return (Ax)_I
    */
    double Ax(const std::vector<double> &u, int i, int j);

    /*!
      \brief Compute the coefficient (Nx)_I

      \param u solution at current iteration
      \param new_u solution at next iteration
      \param i x-component index
      \param j y-component index
      \return (Nx)_I
    */
    double Nx(const std::vector<double> &u, const std::vector<double> &new_u, int i, int j);

    /*!
      \brief Compute the initial squared residual (\|b\|)

      \param source source vector
      \return initial squared residual
    */
    double InitialSquaredResidual(const std::vector<double> &source);

    /*!
      \brief Compute the squared residual

      \param u solution vector
      \param source source vector
      \return squared residual
    */
    double SquaredResidual(const std::vector<double> &u, const std::vector<double> &source);

    /*!
      \brief Compute the squared error

      \param u solution vector
      \param u_exact exact solution vector
      \return squared error
    */
    double SquaredError(const std::vector<double> &u, const std::vector<double> &u_exact);

    /*!
      \brief Compute one step of the Jacobi algorithm

      \param u solution vector
      \param new_u new solution vector
      \param source source vector
    */
    void ComputeOneStep(const std::vector<double> &u, std::vector<double> &new_u, const std::vector<double> &source);

    /*!
      \brief Solve the linear system

      \param max_iterations maximum number of iterations of the Jacobi algorithm
      \param epsilon tolerance for the stopping criteria
      \param verbose (Optional) if true display the residual (and potentatially the error if the exact solution has been provided in the constructor) at each iteration
    */
    void Solve(int max_iterations, double epsilon, bool verbose = false);

    /*!
      \brief Save the given GLOBAL field defined on the whole domain in a file

      \param field considered field (here it is used to save the global solution)
      \param file_name file name 
      \param folder (Optionnal) folder where to save the file (default: ../results/)
      \param format (Optionnal) format of the file (default: .csv)
    */
    void SaveOnDomain(const std::vector<double> &field, std::string file_name, std::string folder = "./results/", std::string format = ".csv");
};

#endif // GAUSS_SEIDEL_SEQUENTIAL_HPP