#!/bin/bash

rm -f ../study_cases/2_jacobi_performances/results/jacobi_strong_scalability_results.csv

echo "Processes; Time" >> ../study_cases/2_jacobi_performances/results/jacobi_strong_scalability_results.csv

echo "Running sequential..."
./2_jacobi_seq_strong_scalability


processes=(1 2 3 4 5 6 7 8 9 10 11 12 13)

for proc in "${processes[@]}"
do
    echo "Running with $proc processes..."
    mpirun -np $proc 2_jacobi_mpi_strong_scalability
done