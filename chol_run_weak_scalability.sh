#!/bin/bash

# Ensure output directory exists
mkdir -p ${SLURM_SUBMIT_DIR}/cholesky_measurements/weak_scalability

# Liste des nombres de processus à tester
procs=(1 2 4 8 16 32)

# Script de soumission pour chaque nombre de processus
for procs_count in "${procs[@]}"
do
    sbatch --job-name="strong_scalability_measurement_$procs_count" \
           --ntasks=$procs_count \
	   --partition=cpu_test \
	   --account=ams301 \
	   --time=00:10:00 \
           --output=${SLURM_SUBMIT_DIR}/cholesky_measurements/weak_scalability/$procs_count.txt \
           --error=${SLURM_SUBMIT_DIR}/cholesky_measurements/weak_scalability/$procs_count.txt \
           --exclusive <<EOF

done
