#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <mpi.h>

// mpicxx jacobi_mpi.cpp -o jacobi_mpi

// Definition of functions pointers 
typedef double (*OneVarFuncPtr)(double);            // of one variable use for boundary condition
typedef double (*TwoVarsFuncPrt)(double, double);   // of 2 variables for domain interior


class JacobiMPI {
public:
    // domain properties
    const int Nx_;  //! number of points within the global domain in the x-direction
    const int Ny_;  //! number of points within the subdomain in the y-direction
    int num_local_rows_; //! number of rows of the subdomain (include the ghost cells)

    const double x_min_;    //! global domain x-direction lower bound
    const double x_max_;    //! global domain x-direction upper bound
    const double y_min_;    //! global domain y-direction lower bound
    const double y_max_;    //! global domain y-direction upper bound

    const double dx_;   //! step size in the x-direction
    const double dy_;   //! step size in the y-direction

    // problem properties
    std::vector<double> local_sol_;       //! current iteration local sol (e.g. x^(l))
    std::vector<double> new_local_sol_;   //! next iteration local sol (e.g. x^(l+1))
    std::vector<double> local_source_;    //! local source term of the PDE
    std::vector<double> local_exact_sol_; //! local exact sol (defined only if the exact sol has been provided)
    std::vector<double> global_sol_;    //! sol on the whole domain

    // solver properties
    double r_init_; //! initial residual (e.g. ||r^(0)||)
    double r_; //! current residual (e.g. ||r^(l)||)
    double error_; //! current error (defined only if the exact sol has been provided)
    int last_l_; //! last iteration (-1 if no convergence)
    std::string class_name_; //! name of the class (used for saving)

    // process properties
    int process_rank_;
    int num_processes_;
    int j_start_;

    // ----------------------------------------------------
    JacobiMPI(
        int Nx, int Ny, double x_min, double x_max, double y_min, double y_max,
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
        OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
        TwoVarsFuncPrt source_func, TwoVarsFuncPrt initi_guess_func,
        TwoVarsFuncPrt exact_sol_func = nullptr, bool verbose = false);

    void InitializeLocalProblem(
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
    OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
    TwoVarsFuncPrt source_func, TwoVarsFuncPrt init_guess_func,
    TwoVarsFuncPrt exact_sol_func);

    void CheckMPIError(int errCode, const std::string &errMsg);

    int I(int i, int j);

    float ComputeLocalSquaredResidual(const std::vector<double> &u, const std::vector<double> &source);
    // ComputeSquaredResidual();

    float ComputeLocalSquaredError(const std::vector<double> &u, const std::vector<double> &u_exact);
    // ComputeLocalSquaredError();

    void ComputeOneLocalStep(const std::vector<double> &u, std::vector<double> &new_u, const std::vector<double> &source);

    // ComputeOneStep();

    void Solve(int max_iteration, double epsilon, bool verbose = false);

    void GatherGlobalSolution();

    void ExchangeBoundaries();

    void SaveOnDomain(const std::vector<double> &field, std::string file_name, std::string format = ".csv");
};

JacobiMPI::JacobiMPI(
    int Nx, int Ny, double x_min, double x_max, double y_min, double y_max,
    OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
    OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
    TwoVarsFuncPrt source_func, TwoVarsFuncPrt init_guess_func,
    TwoVarsFuncPrt exact_sol_func, bool verbose) : Nx_(Nx), Ny_(Ny), x_min_(x_min), x_max_(x_max), y_min_(y_min), y_max_(y_max), dx_((x_max - x_min) / (Nx + 1)), dy_((y_max - y_min) / (Ny + 1)), last_l_(-1), class_name_(__func__) {
    
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank_);

    // Getting rows indices of the subdomain
    int rows_per_process = Ny / num_processes_ + 1;
    j_start_ = process_rank_ * rows_per_process + 1;                // First row of the subdomain
    int j_stop = (process_rank_ + 1) * rows_per_process;               // Last row of the subdomain
    j_stop = (j_stop <= Ny) ? j_stop : Ny;                       // if j_stop > nY => j_stop = nY
    num_local_rows_ = j_stop - j_start_ + 1;                       // number of rows of the subdomain
    int num_local_points = (Nx + 2) * (num_local_rows_ + 2);   // number of points in the subdomain (+2 for the ghost cells)

    // std::cout << "Nx : " << Nx << ", Ny : " << Ny << "\n";
    // std::cout << "j_start_ : " << j_start_ << ", j_stop : " << j_stop << "\n";

    local_sol_.resize(num_local_points, 0.0);
    new_local_sol_.resize(num_local_points, 0.0);
    local_source_.resize(num_local_points, 0.0);
    local_exact_sol_.resize(num_local_points, 0.0);


    // if an exact sol function has been provided
    if (exact_sol_func != nullptr){
        local_exact_sol_.resize(num_local_points, 0.0);
    };

    if (process_rank_ == 0) {
        global_sol_.resize((Nx + 2) * (Ny + 2));
    }

    InitializeLocalProblem(left_condition_func, right_condition_func, top_condition_func, bottom_condition_func, source_func, init_guess_func, exact_sol_func);

    double local_squared_residual_ = ComputeLocalSquaredResidual(local_sol_, local_source_);

    CheckMPIError(
        MPI_Reduce(&local_squared_residual_, &r_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD),
        "Error reducing global squared residual");
    
    if (process_rank_ == 0) {
        r_init_ = std::sqrt(r_);
        if (verbose){
            std::cout << "Problem initialized with following properties : \n";
            std::cout << "| Boundaries : [" << x_min_ << ", " << x_max_ << "]x[" << y_min_ << ", " << y_max_ << "]\n";
            std::cout << "| Grid size : " << Nx_ + 2 << "x" << Ny_ + 2 << "\n";
            std::cout << "| dx : " << dx_ << ", dy : " << dy_ << "\n";
            std::cout << "| r^(0) : " << r_init_ << "\n";
            if (exact_sol_func != nullptr){
                std::cout << "| e^(0) : " << std::sqrt(r_init_) << "\n";
            };
        }
    }
};

void JacobiMPI::CheckMPIError(int errCode, const std::string& errMsg) {
    if (errCode != MPI_SUCCESS) {
        std::cerr << "MPI Error on process " << process_rank_ << " : " << errMsg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, errCode);
    }
};

int JacobiMPI::I(int i, int j){
    return i + j * (Nx_ + 2);
}

// void JacobiMPI::InitializeLocalProblem(
//     OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
//     OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
//     TwoVarsFuncPrt source_func, TwoVarsFuncPrt init_guess_func,
//     TwoVarsFuncPrt exact_sol_func) {
//     for (int i = 0; i <= Nx_ + 1; i++) {
//         double x = x_min_ + i * dx_;
//         if (process_rank_ == 0){    // top and bottom boundary condition of the global sol
//             global_sol_[I(i, Ny_ + 1)] = top_condition_func(x);
//             global_sol_[I(i, 0)] = bottom_condition_func(x);
//         }

//         for (int j = 0; j <= num_local_rows_ + 1; j++)
//         {
//             double y = y_min_ + (j + j_start_ - 1) * dy_;

//             if (i == 0)
//             {
//                 local_sol_[I(i, j)] = new_local_sol_[I(i, j)] = left_condition_func(x);
//             }
//             else if (i == Nx_ + 1){
//                 local_sol_[I(i, j)] = new_local_sol_[I(i, j)] = right_condition_func(x);
//             }
//             else if (j == 0){
//                 local_sol_[I(i, j)] = new_local_sol_[I(i, j)] = init_guess_func(x, y);
//                 if (process_rank_ == 0){
//                     // local_sol_[I(i, j)] = new_local_sol_[I(i, j)] = bottom_condition_func(y);
//                     local_source_[I(i, j)] = source_func(x, y);
//                     if (exact_sol_func != nullptr){
//                         local_exact_sol_[I(i, j)] = exact_sol_func(x, y);
//                     }
//                 }
//             }
//             else if (j == num_local_rows_ + 1){
//                 local_sol_[I(i, j)] = new_local_sol_[I(i, j)] = init_guess_func(x, y);
//                 if (process_rank_ == num_processes_ - 1){
//                     // local_sol_[I(i, j)] = new_local_sol_[I(i, j)] = top_condition_func(y);
//                     local_source_[I(i, j)] = source_func(x, y);
//                     if (exact_sol_func != nullptr){
//                         local_exact_sol_[I(i, j)] = exact_sol_func(x, y);
//                     }
//                 }
//             }
//             else {
//                 local_sol_[I(i, j)] = new_local_sol_[I(i, j)] = init_guess_func(x, y);
//                 local_source_[I(i, j)] = source_func(x, y);
//                 if (exact_sol_func != nullptr){
//                     local_exact_sol_[I(i, j)] = exact_sol_func(x, y);
//                 }
//             }

            
//         }
//     }
// };

void JacobiMPI::InitializeLocalProblem(
    OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
    OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
    TwoVarsFuncPrt source_func, TwoVarsFuncPrt init_guess_func,
    TwoVarsFuncPrt exact_sol_func) {
    for (int i = 0; i <= Nx_ + 1; i++) {
        double x = x_min_ + i * dx_;
        if (process_rank_ == 0){    // top and bottom boundary condition of the global sol
            global_sol_[I(i, Ny_ + 1)] = top_condition_func(x);
            global_sol_[I(i, 0)] = bottom_condition_func(x);
        }

        for (int j = 0; j <= num_local_rows_ + 1; j++)
        {
            double y = y_min_ + (j + j_start_ - 1) * dy_;
            
            if (i == 0)
            {
                local_sol_[I(i, j)] = new_local_sol_[I(i, j)] = left_condition_func(x);
            }
            else if (i == Nx_ + 1){
                local_sol_[I(i, j)] = new_local_sol_[I(i, j)] = right_condition_func(x);
            }
            else {
                local_sol_[I(i, j)] = new_local_sol_[I(i, j)] = init_guess_func(x, y);
                local_source_[I(i, j)] = source_func(x, y);
                if (exact_sol_func != nullptr){
                        local_exact_sol_[I(i, j)] = exact_sol_func(x, y);
                }
            }

        }
    }
};


float JacobiMPI::ComputeLocalSquaredResidual(const std::vector<double> &u, const std::vector<double> &source){
    double value = 0.0;
    double dx2 = dx_ * dx_;
    double dy2 = dy_ * dy_;
    for (int i = 1; i <= Nx_; i++)
    {
        for (int j = 1; j <= num_local_rows_; j++){
            double tmp = source[I(i, j)] - ( (u[I(i+1,j)] - 2.*u[I(i,j)] + u[I(i-1,j)])/dx2 + (u[I(i,j+1)] - 2.*u[I(i,j)] + u[I(i,j-1)])/dy2);
            value += tmp * tmp;
        }
    }
    return value;
};


// float JacobiMPI::ComputeLocalSquaredError(const std::vector<double> &u, const std::vector<double> &u_exact){
//     double value = 0.0;
//     for (int i = 0; i <= Nx_ + 1; i++)
//     {
//         for (int j = 0; j <= num_local_rows_ + 1; j++){
//             double tmp = u_exact[I(i, j)] - u[I(i, j)];
//             value += tmp * tmp;
//         }
//     }
//     return value;
// };
float JacobiMPI::ComputeLocalSquaredError(const std::vector<double> &u, const std::vector<double> &u_exact){
    double value = 0.0;
    // MPI_Barrier(MPI_COMM_WORLD);
    
    // int j_start = (1 - process_rank_ == 0);
    int j_start = (process_rank_ == 0) ? 0 : 1;
    
    // int j_end = num_local_rows_ + (process_rank_ == num_processes_ - 1);
    int j_end = (process_rank_ == num_processes_ - 1) ? num_local_rows_ + 1 : num_local_rows_;
    // std::cout << "process_rank_ :" << process_rank_ << ", process_rank_ == 0 : " << (process_rank_ == 0) << ", process_rank_ == num_processes_ - 1 : " << (process_rank_ == num_processes_ - 1)<< "j_start :" << j_start << ", j_end : " << j_end << "\n";
    // std::cout << "num_local_rows_ :" << num_local_rows_ << "\n";
    // std::cout << "j_start :" << j_start << ", j_end : " << j_end << "\n";
    for (int i = 1; i <= Nx_; i++)
    {
        // for (int j = j_start; j <= j_end; j++){
        for (int j = j_start; j <= j_end; j++){
            double tmp = u_exact[I(i, j)] - u[I(i, j)];
            value += tmp * tmp;
        }
    }
    return value;
};


void JacobiMPI::ComputeOneLocalStep(const std::vector<double>& u, std::vector<double>& new_u, const std::vector<double>& source) {
    double dx2 = dx_ * dx_;
    double dy2 = dy_ * dy_;
    double A_ii = -2. * (1. / dx2 + 1. / dy2);
    for (int i = 1; i <= Nx_; i++) {
        for (int j = 1; j <= num_local_rows_; j++) {
            new_u[I(i, j)] = (1. / A_ii) * (- (u[I(i + 1, j)] + u[I(i - 1, j)]) / dx2 - (u[I(i, j + 1)] + u[I(i, j - 1)]) / dy2 + source[I(i, j)]);
        };
    };
};


void JacobiMPI::Solve(int max_iterations, double epsilon, bool verbose) {
    bool converged = false;
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    ExchangeBoundaries(); // just to be sure that every process has the correct initialization
    for (int l = 1; l <= max_iterations; l++)
    {
        if (converged) break;
        ComputeOneLocalStep(local_sol_, new_local_sol_, local_source_);
        local_sol_.swap(new_local_sol_);
        ExchangeBoundaries();

        // every process compute its local residual and sends it to process 0
        double local_squared_residual_ = ComputeLocalSquaredResidual(local_sol_, local_source_);
        CheckMPIError(
            MPI_Reduce(&local_squared_residual_, &r_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD),
            "Error reducing global squared residual");


        if (process_rank_ == 0) {
            r_ = std::sqrt(r_);

            if (verbose){
                std::cout << "l : " << l << ", last residual : " << r_;
                if (!local_exact_sol_.empty()) {
                    double local_squared_error = ComputeLocalSquaredError(local_sol_, local_exact_sol_);
                    CheckMPIError(MPI_Reduce(&local_squared_error, &error_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD), "Error reducing global squared error");  // every process sends to the process 0 its local error

                    std::cout << ", error :  " << std::sqrt(error_);
                }
                std::cout << "\n";
            }

            if (r_ <= epsilon * r_init_) {
                std::cout << "Converged after " << l << " iterations, last residual : " << r_ << "\n";
                last_l_ = l;
                converged = true;
                // break;
            }
        }
        // Every process check if the algorithm converged
        CheckMPIError(
                MPI_Bcast(&converged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD),
                "Error broadcasting convergence signal"); // process 0 sends to every process if the algorithm has converged
    }


    if ((last_l_ == -1) and (process_rank_ == 0)){
        std::cout << "No convergence, last residual : " << r_ << "\n";
    }

    if (!local_exact_sol_.empty()) {
        double local_squared_error = ComputeLocalSquaredError(local_sol_, local_exact_sol_);
        CheckMPIError(MPI_Reduce(&local_squared_error, &error_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD), "Error reducing global squared error");  // every process sends to the process 0 its local error

        if (process_rank_ == 0){
            std::cout << "Last error :  " << std::sqrt(error_) << "\n";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    if (process_rank_ == 0) {
        std::cout << "Total execution time: " << end_time - start_time << " seconds" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    GatherGlobalSolution();
    if (process_rank_ == 0){
        SaveOnDomain(global_sol_, "last_iteration_solution");
    }
}


void JacobiMPI::GatherGlobalSolution() {
    MPI_Gather(
        &local_sol_[I(0, 1)],          // starting address of send buffer
        (Nx_ + 2) * num_local_rows_, // number of elements in send buffer
        MPI_DOUBLE,                         // data type of send buffer elements
        &global_sol_[I(0, 1)],         // Starting address of receive buffer
        (Nx_ + 2) * num_local_rows_, // number of elements for any single receive
        MPI_DOUBLE,                         // data type of recv buffer elements
        0,                                  // rank of receiving process
        MPI_COMM_WORLD);                    // communicator
};

void JacobiMPI::ExchangeBoundaries() {
    MPI_Request req_send_bottom, req_recv_top;
    if (process_rank_ > 0)
    { // => the subdomain has a neighbourd at its bottom
        MPI_Isend(
            &local_sol_[I(0, 1)], // Include all points from left to right
            Nx_ + 2,            // Number of points along the x-direction
            MPI_DOUBLE,
            process_rank_ - 1,
            0,
            MPI_COMM_WORLD,
            &req_send_bottom);

        MPI_Irecv(
            &local_sol_[I(0, 0)], // Include ghost cells for the bottom row
            Nx_ + 2,
            MPI_DOUBLE,
            process_rank_ - 1,
            0,
            MPI_COMM_WORLD,
            &req_recv_top);
    }

    MPI_Request req_send_top, req_recv_bottom;
    if (process_rank_ < num_processes_ - 1)
    { // => the subdomain has a neighbourd at its top
        MPI_Isend(
            &local_sol_[I(0, num_local_rows_)], // top line
            Nx_ + 2,
            MPI_DOUBLE,
            process_rank_ + 1,
            0,
            MPI_COMM_WORLD,
            &req_send_top); // send the last line of the subdomain to its top neighbour
        MPI_Irecv(
            &local_sol_[I(0, num_local_rows_ + 1)],
            Nx_ + 2,
            MPI_DOUBLE,
            process_rank_ + 1,
            0,
            MPI_COMM_WORLD,
            &req_recv_bottom); // receive the first line of the subdomain' top neighbour
    }

    if (process_rank_ > 0) {
        MPI_Wait(&req_send_bottom, MPI_STATUS_IGNORE);
        MPI_Wait(&req_recv_top, MPI_STATUS_IGNORE);
    }
    if (process_rank_ < num_processes_ - 1) {
        MPI_Wait(&req_send_top, MPI_STATUS_IGNORE);
        MPI_Wait(&req_recv_bottom, MPI_STATUS_IGNORE);
    }
};

void JacobiMPI::SaveOnDomain(const std::vector<double> &field, std::string file_name, std::string format) {
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
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    int n = 100;
    double x_min = 0.;
    double y_min = 0.;
    double x_max = a;
    double y_max = b;
    // JacobiMPI jac(n, n )
    MPI_Barrier(MPI_COMM_WORLD);
    double init_start = MPI_Wtime();
    JacobiMPI jac_mpi(
        n, n, x_min, x_max, y_min, y_max, 
        dirichlet_condition, dirichlet_condition, dirichlet_condition, dirichlet_condition, 
        laplacien_sin_sin, constant, sin_sin, false);

    double init_end = MPI_Wtime();  // Temps de fin de l'initialisation

    if (process_rank == 0) {
        std::cout << "Initialization time : " << init_end - init_start << " secondes" << std::endl;
    }

    jac_mpi.Solve(20000, 1e-6);

    MPI_Barrier(MPI_COMM_WORLD);
    double total_end = MPI_Wtime();

    if (process_rank == 0) {
        std::cout << "Total time : " << total_end - init_start << " secondes" << std::endl;
    }

    MPI_Finalize();
    return 0;
};

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
//     // JacobiMPI jac(n, n )


//     // --------------------------------
//     int n_min = 50; 
//     int n_max = 350; 
//     // double epsilon_min = 1e-1; 
//     double epsilon_min = 1e-4;
//     double epsilon_max = 1e-5;
//     int maxIterations = 100000;
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
//             JacobiMPI jac_seq(
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