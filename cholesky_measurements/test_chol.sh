#!/bin/bash
#SBATCH --job-name=mpi_job_1
#SBATCH --partition=cpu_test
#SBATCH --account=ams301
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --output=mpi_job_1.txt
#SBATCH --error=mpi_job_1.txt

# Load modules
module load cmake
module load gcc
module load gmsh
module load openmpi

# Execution
mpirun -np ${SLURM_NTASKS} ${SLURM_SUBMIT_DIR}/2_strong_scalability
