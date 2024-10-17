#!/bin/bash

rm -f ../study_cases/2_strong_scalability/jacobi_strong_scalability.csv
rm -f ../study_cases/2_strong_scalability/gauss_seidel_strong_scalability.csv

echo "Processes; Time" >> ../study_cases/2_strong_scalability/jacobi_strong_scalability.csv
echo "Processes; Time" >> ../study_cases/2_strong_scalability/gauss_seidel_strong_scalability.csv

processes=(1 2 3 4 5 6 7 8 9 10 11 12 13)

for proc in "${processes[@]}"
do
    echo "Running with $proc processes..."
    mpirun -np $proc 2_strong_scalability
done