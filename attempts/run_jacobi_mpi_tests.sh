#!/bin/bash

# Crée le répertoire des résultats s'il n'existe pas
mkdir -p results

# Supprime le fichier de résultats existant
rm -f ../study_cases/2_jacobi_performances/results/mpi_performance_results.csv

# Ajoute l'en-tête du fichier CSV
echo "Processes; Time" >> ../study_cases/2_jacobi_performances/results/mpi_performance_results.csv

# Liste des nombres de processus à tester
processes=(1 2 3 4 5 6 7 8 9 10 12)

# Lancer les tests pour différents nombres de processus
for proc in "${processes[@]}"
do
    echo "Running with $proc processes..."
    # mpirun -np $proc ${CMAKE_BINARY_DIR}/2_jacobi_mpi_perf
    mpirun -np $proc 2_jacobi_mpi_perf
done