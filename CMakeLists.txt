# Specify the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name
project(ITERATIVE_METHODS)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the MPI package, required for parallel processing
find_package(MPI REQUIRED)

# Define the source directory for the project
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Gather all .cpp files from the 'solvers' directory
file(GLOB solvers_srcs ${SOURCE_DIR}/solvers/*.cpp)

# Include directories for headers
include_directories(${SOURCE_DIR}/solvers)
include_directories(${MPI_INCLUDE_PATH})

# Create the main executable
file(GLOB main_src ${SOURCE_DIR}/main.cpp)
add_executable(main ${main_src} ${solvers_srcs})
target_link_libraries(main MPI::MPI_CXX)  # Link MPI library to main
target_compile_options(main PRIVATE -O2)   # Set compiler optimization level

# Case study 1: Jacobi Sequential Asymptotic Accuracy
file(GLOB case1_srcs ${SOURCE_DIR}/study_cases/1_jacobi_seq_asymptotic_accuracy/1_jacobi_seq_asymptotic_accuracy.cpp)
add_executable(1_jacobi_seq_asymptotic_accuracy ${case1_srcs} ${solvers_srcs})
target_link_libraries(1_jacobi_seq_asymptotic_accuracy MPI::MPI_CXX)  # Link MPI library
target_compile_options(1_jacobi_seq_asymptotic_accuracy PRIVATE -O2)   # Set optimization

# Case study 2: Jacobi Performance Measurements
file(GLOB jacobi_strong_scalability_mpi_srcs ${SOURCE_DIR}/study_cases/2_jacobi_performances/2_jacobi_mpi_strong_scalability.cpp)
add_executable(2_jacobi_mpi_strong_scalability ${jacobi_strong_scalability_mpi_srcs} ${solvers_srcs})
target_link_libraries(2_jacobi_mpi_strong_scalability MPI::MPI_CXX)  # Link MPI library
target_compile_options(2_jacobi_mpi_strong_scalability PRIVATE -O2)   # Set optimization

# Add permission to execute the strong scalability MPI executable
add_custom_command(TARGET 2_jacobi_mpi_strong_scalability POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Giving execute permission to 2_jacobi_mpi_strong_scalability"
    COMMAND chmod +x $<TARGET_FILE:2_jacobi_mpi_strong_scalability>
    COMMENT "Giving execute permission to the 2_jacobi_mpi_strong_scalability executable"
)

# Strong scalability with sequential Jacobi method
file(GLOB jacobi_strong_scalability_seq_srcs ${SOURCE_DIR}/study_cases/2_jacobi_performances/2_jacobi_seq_strong_scalability.cpp)
add_executable(2_jacobi_seq_strong_scalability ${jacobi_strong_scalability_seq_srcs} ${solvers_srcs})
target_link_libraries(2_jacobi_seq_strong_scalability MPI::MPI_CXX)  # Link MPI library
target_compile_options(2_jacobi_seq_strong_scalability PRIVATE -O2)   # Set optimization

# Add permission to execute the strong scalability sequential executable
add_custom_command(TARGET 2_jacobi_seq_strong_scalability POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Giving execute permission to 2_jacobi_seq_strong_scalability"
    COMMAND chmod +x $<TARGET_FILE:2_jacobi_seq_strong_scalability>
    COMMENT "Giving execute permission to the 2_jacobi_seq_strong_scalability executable"
)

# Custom target to run the MPI strong scalability measurements
add_custom_target(run_jacobi_strong_scalability_measurements
    COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_CURRENT_SOURCE_DIR}/study_cases/2_jacobi_performances/run_jacobi_strong_scalability.sh
    DEPENDS 2_jacobi_mpi_strong_scalability 2_jacobi_seq_strong_scalability
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}  # Set the working directory
    COMMENT "Executing the Bash script to test the Jacobi strong scalability"
)

# Weak scalability case study
file(GLOB jacobi_weak_scalability_srcs ${SOURCE_DIR}/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp)
add_executable(2_jacobi_weak_scalability ${jacobi_weak_scalability_srcs} ${solvers_srcs})
target_link_libraries(2_jacobi_weak_scalability MPI::MPI_CXX)  # Link MPI library
target_compile_options(2_jacobi_weak_scalability PRIVATE -O2)   # Set optimization

# Add permission to execute the weak scalability executable
add_custom_command(TARGET 2_jacobi_weak_scalability POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Giving execute permission to jacobi_weak_scalability"
    COMMAND chmod +x $<TARGET_FILE:2_jacobi_weak_scalability>
    COMMENT "Giving execute permission to the jacobi_weak_scalability executable"
)

# Custom target to run the weak scalability measurements
add_custom_target(run_jacobi_weak_scalability_measurements
    COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_CURRENT_SOURCE_DIR}/study_cases/2_jacobi_performances/run_jacobi_weak_scalability.sh
    DEPENDS 2_jacobi_weak_scalability
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}  # Set the working directory
    COMMENT "Executing the Bash script to test the Jacobi weak scalability"
)

# Enable exporting of compilation commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")