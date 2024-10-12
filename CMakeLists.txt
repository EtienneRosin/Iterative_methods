cmake_minimum_required(VERSION 3.10)

# Nom du projet
project(ITERATIVE_METHODS)

# Spécifier le standard C++ (par exemple, C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Ajouter la recherche de MPI
find_package(MPI REQUIRED)

# Définir les répertoires de sources
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Récupérer tous les fichiers .cpp dans les sous-dossiers utils et solvers
file(GLOB solvers_srcs ${SOURCE_DIR}/solvers/*.cpp)

# Inclure les répertoires pour trouver les fichiers headers
include_directories(${SOURCE_DIR}/solvers)
include_directories(${MPI_INCLUDE_PATH})

# Créer l'exécutable principal
file(GLOB main_src ${SOURCE_DIR}/main.cpp)
add_executable(main ${main_src} ${solvers_srcs})
target_link_libraries(main MPI::MPI_CXX)
target_compile_options(main PRIVATE -O2)

### 2. Cas d'étude 1 : jacobi_sequential_asymptotic_accuracy --------------------------------------------------------------------
file(GLOB case1_srcs ${SOURCE_DIR}/study_cases/1_jacobi_seq_asymptotic_accuracy/1_jacobi_seq_asymptotic_accuracy.cpp)
add_executable(1_jacobi_seq_asymptotic_accuracy ${case1_srcs} ${solvers_srcs})
target_link_libraries(1_jacobi_seq_asymptotic_accuracy MPI::MPI_CXX)
target_compile_options(1_jacobi_seq_asymptotic_accuracy PRIVATE -O2)

### 3. Cas d'étude 2 : Jacobi Performances --------------------------------------------------------------------------------------
file(GLOB jacobi_strong_scalability_mpi_srcs ${SOURCE_DIR}/study_cases/2_jacobi_performances/2_jacobi_mpi_strong_scalability.cpp)
add_executable(2_jacobi_mpi_strong_scalability ${jacobi_strong_scalability_mpi_srcs} ${solvers_srcs})
target_link_libraries(2_jacobi_mpi_strong_scalability MPI::MPI_CXX)
target_compile_options(2_jacobi_mpi_strong_scalability PRIVATE -O2)
    # allow permission
add_custom_command(TARGET 2_jacobi_mpi_strong_scalability POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Giving execute permission to 2_jacobi_mpi_strong_scalability"
    COMMAND chmod +x $<TARGET_FILE:2_jacobi_mpi_strong_scalability>
    COMMENT "Giving execute permission to the 2_jacobi_mpi_strong_scalability executable"
)

file(GLOB jacobi_strong_scalability_seq_srcs ${SOURCE_DIR}/study_cases/2_jacobi_performances/2_jacobi_seq_strong_scalability.cpp)
add_executable(2_jacobi_seq_strong_scalability ${jacobi_strong_scalability_seq_srcs} ${solvers_srcs})
target_link_libraries(2_jacobi_seq_strong_scalability MPI::MPI_CXX)  
target_compile_options(2_jacobi_seq_strong_scalability PRIVATE -O2)

add_custom_command(TARGET 2_jacobi_seq_strong_scalability POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Giving execute permission to 2_jacobi_seq_strong_scalability"
    COMMAND chmod +x $<TARGET_FILE:2_jacobi_seq_strong_scalability>
    COMMENT "Giving execute permission to the 2_jacobi_seq_strong_scalability executable"
)

# Cible personnalisée pour lancer les tests MPI
add_custom_target(run_jacobi_strong_scalability_measurements
    COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_CURRENT_SOURCE_DIR}/study_cases/2_jacobi_performances/run_jacobi_strong_scalability.sh
    DEPENDS 2_jacobi_mpi_strong_scalability 2_jacobi_seq_strong_scalability
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Executing the Bash script to test the Jacobi strong scalability"
)


# --------------------------
file(GLOB jacobi_weak_scalability_srcs ${SOURCE_DIR}/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp)
add_executable(2_jacobi_weak_scalability ${jacobi_weak_scalability_srcs} ${solvers_srcs})
target_link_libraries(2_jacobi_weak_scalability MPI::MPI_CXX)
target_compile_options(2_jacobi_weak_scalability PRIVATE -O2)

add_custom_command(TARGET 2_jacobi_weak_scalability POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Giving execute permission to jacobi_weak_scalability_srcs"
    COMMAND chmod +x $<TARGET_FILE:2_jacobi_weak_scalability>
    COMMENT "Giving execute permission to the jacobi_weak_scalability_srcs executable"
)

add_custom_target(run_jacobi_weak_scalability_measurements
    COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_CURRENT_SOURCE_DIR}/study_cases/2_jacobi_performances/run_jacobi_weak_scalability.sh
    DEPENDS 2_jacobi_weak_scalability
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Executing the Bash script to test the Jacobi weak scalability"
)

# Exporter les commandes de compilation
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")