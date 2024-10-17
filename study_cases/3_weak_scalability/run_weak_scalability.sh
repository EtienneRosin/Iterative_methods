#!/bin/bash

rm -f ../study_cases/3_weak_scalability/jacobi_weak_scalability.csv
rm -f ../study_cases/3_weak_scalability/gauss_seidel_weak_scalability.csv

echo "Processes; Time" >> ../study_cases/3_weak_scalability/jacobi_weak_scalability.csv
echo "Processes; Time" >> ../study_cases/3_weak_scalability/gauss_seidel_weak_scalability.csv

processes=(1 2 3 4 5 6 7 8 9 10 11 12 13)

for proc in "${processes[@]}"
do
    echo "Running with $proc processes..."
    mpirun -np $proc 3_weak_scalability
done