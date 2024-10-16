// mpicxx main.cpp -o main
// mpirun -np 2 main

#include "jacobi_mpi.hpp"

using OneVarFuncPtr = utils::OneVarFuncPtr;
using TwoVarsFuncPtr = utils::TwoVarsFuncPtr;

JacobiMPI::JacobiMPI(
        int Nx, int Ny, double x_min, double x_max, double y_min, double y_max,
        OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
        OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
        TwoVarsFuncPtr source_func, TwoVarsFuncPtr init_guess_func,
        TwoVarsFuncPtr exact_sol_func, bool verbose) 
        : Nx_(Nx), Ny_(Ny), 
        x_min_(x_min), x_max_(x_max), 
        y_min_(y_min), y_max_(y_max), 
        dx_((x_max - x_min) / (Nx + 1)), dy_((y_max - y_min) / (Ny + 1)), 
        dx2_(dx_ * dx_), dy2_(dy_ * dy_), last_l_(-1), class_name_(__func__) {
    
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank_);

    // Getting rows indices of the subdomain
    int rows_per_process = Ny / num_processes_;
    j_start_ = process_rank_ * rows_per_process + 1;
    int j_stop = (process_rank_ + 1) * rows_per_process 
        + (Ny % num_processes_) * (process_rank_ == num_processes_ - 1);
    num_local_rows_ = j_stop - j_start_ + 1;
    int num_local_points = (Nx + 2) * (num_local_rows_ + 2);

    // Resizing the local fields 
    local_sol_.resize(num_local_points, 0.0);
    new_local_sol_.resize(num_local_points, 0.0);
    local_source_.resize(num_local_points, 0.0);
    local_exact_sol_.resize(num_local_points, 0.0);

    // if an exact solution function has been provided
    if (exact_sol_func != nullptr){
        local_exact_sol_.resize(num_local_points, 0.0);
    };

    // Resizing the global solution
    if (process_rank_ == 0) {
        global_sol_.resize((Nx + 2) * (Ny + 2));
    }

    // Initializing the local problem
    InitializeLocalProblem(
        left_condition_func, right_condition_func, 
        top_condition_func, bottom_condition_func, 
        source_func, init_guess_func, exact_sol_func);

    double local_squared_residual_ = LocalInitialSquaredResidual(local_source_);

    CheckMPIError(
        MPI_Reduce(
            &local_squared_residual_, 
            &r_, 
            1, 
            MPI_DOUBLE, 
            MPI_SUM, 
            0, MPI_COMM_WORLD),
        "Error reducing global squared residual");
    
    if (exact_sol_func != nullptr){
        double local_squared_error = LocalSquaredError(local_sol_, local_exact_sol_);
        CheckMPIError(
            MPI_Reduce(
                &local_squared_error, 
                &error_, 
                1, 
                MPI_DOUBLE, 
                MPI_SUM, 
                0, 
                MPI_COMM_WORLD),
            "Error reducing global squared error");
    };
    
    if (process_rank_ == 0) {
        r_init_ = std::sqrt(r_);


        if (verbose){
            std::cout << "Problem initialized with following properties : \n";
            std::cout << "| Boundaries : [" << x_min_ << ", " 
                      << x_max_ << "]x[" << y_min_ << ", " << y_max_ << "]\n";
            std::cout << "| Grid size : " << Nx_ + 2 << "x" << Ny_ + 2 << "\n";
            std::cout << "| dx : " << dx_ << ", dy : " << dy_ << "\n";
            std::cout << "| r^(0) : " << r_init_ << "\n";
            if (exact_sol_func != nullptr){
                std::cout << "| e^(0) : " << std::sqrt(error_) << "\n";
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


void JacobiMPI::InitializeLocalProblem(
    OneVarFuncPtr left_condition_func, OneVarFuncPtr right_condition_func,
    OneVarFuncPtr top_condition_func, OneVarFuncPtr bottom_condition_func,
    TwoVarsFuncPtr source_func, TwoVarsFuncPtr init_guess_func,
    TwoVarsFuncPtr exact_sol_func) {
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
            }
            local_source_[I(i, j)] = source_func(x, y);
            if (exact_sol_func != nullptr){
                    local_exact_sol_[I(i, j)] = exact_sol_func(x, y);
            }
        }
    }
};


int JacobiMPI::I(int i, int j)
{
    return i + j * (Nx_ + 2);
}

double JacobiMPI::Ax(const std::vector<double> &u, int i, int j){
    return (u[I(i + 1, j)] - 2. * u[I(i, j)] + u[I(i - 1, j)]) / dx2_ + (u[I(i, j + 1)] - 2. * u[I(i, j)] + u[I(i, j - 1)]) / dy2_;
}

double JacobiMPI::Nx(const std::vector<double> &u, int i, int j) {
    return -(u[I(i + 1, j)] + u[I(i - 1, j)]) / dx2_ - (u[I(i, j + 1)] + u[I(i, j - 1)]) / dy2_;
}

double JacobiMPI::LocalSquaredResidual(const std::vector<double> &u, const std::vector<double> &source){
    double value = 0.0;
    for (int i = 1; i <= Nx_; i++)
    {
        for (int j = 1; j <= num_local_rows_; j++){
            double tmp = source[I(i, j)] - Ax(u, i, j);
            value += tmp * tmp;
        }
    }
    return value / ((Nx_ + 2)*(Ny_ + 2));
}

double JacobiMPI::LocalInitialSquaredResidual(const std::vector<double> &source){
    double value = 0.0;
    for (int i = 1; i <= Nx_; i++)
    {
        for (int j = 1; j <= num_local_rows_; j++){
            double tmp = source[I(i, j)];
            value += tmp * tmp;
        }
    }
    return value/ ((Nx_ + 2)*(Ny_ + 2));
}

double JacobiMPI::LocalSquaredError(const std::vector<double> &u, const std::vector<double> &u_exact){
    double value = 0.0;
    int j_start = (process_rank_ == 0) ? 0 : 1;
    int j_end = (process_rank_ == num_processes_ - 1) ? num_local_rows_ + 1 : num_local_rows_;
    for (int i = 0; i <= Nx_ + 1; i++)
    {
        for (int j = j_start; j <= j_end; j++)
        {
            double tmp = u_exact[I(i, j)] - u[I(i, j)];
            value += tmp * tmp;
        }
    }
    return value / ((Nx_ + 2)*(Ny_ + 2));
}

void JacobiMPI::ComputeOneLocalStep(const std::vector<double> &u, std::vector<double> &new_u, const std::vector<double> &source){
    double A_ii_minus_one = - 0.5 * (dx2_ * dy2_) / (dx2_ + dy2_);
    for (int i = 1; i <= Nx_; i++)
    {
        for (int j = 1; j <= num_local_rows_; j++) {
            new_u[I(i, j)] = A_ii_minus_one * (source[I(i, j)] + Nx(u, i, j));
        };
    };
}

void JacobiMPI::Solve(int max_iterations, double epsilon, bool verbose) {
    bool converged = false;
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    ExchangeBoundaries(); // just to be sure that every process has the correct initialization

    for (int l = 1; l <= max_iterations; l++) {
        if (converged) break;
        ComputeOneLocalStep(local_sol_, new_local_sol_, local_source_);
        local_sol_.swap(new_local_sol_);
        ExchangeBoundaries();

        // every process compute its local residual and sends it to process 0
        double local_squared_residual_ = LocalSquaredResidual(local_sol_, local_source_);
        CheckMPIError(
            MPI_Reduce(&local_squared_residual_, &r_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD),
            "Error reducing global squared residual");


        if (process_rank_ == 0) {
            r_ = std::sqrt(r_);

            if (verbose){
                std::cout << "l : " << l << ", last residual : " << r_;
                if (!local_exact_sol_.empty()) {
                    double local_squared_error = LocalSquaredError(local_sol_, local_exact_sol_);
                    CheckMPIError(MPI_Reduce(&local_squared_error, &error_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD), "Error reducing global squared error");  // every process sends to the process 0 its local error

                    std::cout << ", error :  " << std::sqrt(error_);
                }
                std::cout << "\n";
            }

            if (r_ <= epsilon * r_init_) {
                std::cout << "Converged after " << l << " iterations, last residual : " << r_ << "\n";
                last_l_ = l;
                converged = true;
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
        double local_squared_error = LocalSquaredError(local_sol_, local_exact_sol_);
        CheckMPIError(MPI_Reduce(&local_squared_error, &error_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD), "Error reducing global squared error");  // every process sends to the process 0 its local error
        // std::cout << "getting error \n";
        if (process_rank_ == 0)
        {
            // if (verbose){
            // std::cout << "Last error :  " << std::sqrt(error_) << "\n";
            // }
            std::cout << "Last error :  " << std::sqrt(error_) << "\n";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    if (process_rank_ == 0) {
        std::cout << "Total execution time: " << end_time - start_time << " seconds" << std::endl;
    }
    // MPI_Barrier(MPI_COMM_WORLD);
    GatherGlobalSolution();
    if (process_rank_ == 0){
        if (verbose){
            SaveOnDomain(global_sol_, "last_iteration_solution");
        } 
    }
};

void JacobiMPI::GatherGlobalSolution() {
    std::vector<int> recvcounts(num_processes_);
    std::vector<int> displs(num_processes_);

    // Process 0 needs to know the sizes and displacements
    if (process_rank_ == 0) {
        int disp = 0;
        int rows_per_process = Ny_ / num_processes_;
        int j_start;
        int j_stop;
        int num_local_rows_;
        for (int p = 0; p < num_processes_; ++p)
        {
            j_start = p * rows_per_process + 1;
            j_stop = (p + 1) * rows_per_process + (Ny_ % num_processes_) * (p == num_processes_ - 1);
            num_local_rows_ = j_stop - j_start_ + 1;
            recvcounts[p] = (Nx_ + 2) * num_local_rows_;
            displs[p] = j_start - 1;
        }
    }

    // Gather the local solutions into the global solution
    CheckMPIError(
        MPI_Gatherv(
            &local_sol_[I(1, 1)],          // starting address of send buffer (skip ghost cells)
            (Nx_ + 2) * num_local_rows_,   // number of elements in local buffer
            MPI_DOUBLE,                    // data type
            &global_sol_[0],               // starting address of receive buffer (on rank 0)
            recvcounts.data(),             // array of counts (number of elements per process)
            displs.data(),                 // array of displacements
            MPI_DOUBLE,                    // data type
            0,                             // rank of receiving process (root)
            MPI_COMM_WORLD),               // communicator
        "Error gathering the global solution"
    );
}


void JacobiMPI::ExchangeBoundaries() {
    if (process_rank_ > 0) {
        MPI_Sendrecv(
            &local_sol_[I(0, 1)],
            Nx_ + 2,
            MPI_DOUBLE,
            process_rank_ - 1,
            0,
            &local_sol_[I(0, 0)],
            Nx_ + 2,
            MPI_DOUBLE,
            process_rank_ - 1,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    }

    if (process_rank_ < num_processes_ - 1) {
        MPI_Sendrecv(
            &local_sol_[I(0, num_local_rows_)],
            Nx_ + 2,
            MPI_DOUBLE,
            process_rank_ + 1,
            0,
            &local_sol_[I(0, num_local_rows_ + 1)],
            Nx_ + 2,
            MPI_DOUBLE,
            process_rank_ + 1,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    }
}


void JacobiMPI::SaveOnDomain(const std::vector<double> &field, std::string file_name, std::string folder, std::string format){
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