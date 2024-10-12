#!/bin/bash

# Supprime le fichier de résultats existant
rm -f ../study_cases/2_jacobi_performances/results/jacobi_strong_scalability_results.csv

# Ajoute l'en-tête du fichier CSV
echo "Processes; Time" >> ../study_cases/2_jacobi_performances/results/jacobi_strong_scalability_results.csv

echo "Running sequential..."
./2_jacobi_seq_strong_scalability

# Liste des nombres de processus à tester
processes=(1 2 3 4 5 6 8 9 10 12)

# Lancer les tests pour différents nombres de processus
for proc in "${processes[@]}"
do
    echo "Running with $proc processes..."
    # mpirun -np $proc ${CMAKE_BINARY_DIR}/2_jacobi_mpi_perf
    mpirun -np $proc 2_jacobi_mpi_strong_scalability
done