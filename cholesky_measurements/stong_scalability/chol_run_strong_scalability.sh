#!/bin/bash

# Liste des nombres de processus à tester
procs=(1 2 4 8 16)

# Script de soumission pour chaque nombre de processus
for procs_count in "${procs[@]}"
do
    sbatch --job-name="strong_scalability_measurement_$procs_count" \
           --ntasks=$procs_count \
           --output=mpi_job_$procs_count.txt \
           --error=mpi_job_$procs_count.txt \
           --exclusive <<EOF
#!/bin/bash
#SBATCH --partition=cpu_test
#SBATCH --account=ams301
#SBATCH --time=00:10:00
#SBATCH --ntasks=$procs_count

# Load modules
module load cmake
module load gcc
module load gmsh
module load openmpi

# Execution
mpirun -np $procs_count ${SLURM_SUBMIT_DIR}/2_strong_scalability
EOF
done
