#!/bin/bash
#SBATCH --job-name=my_python_job
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB

# Load Anaconda module
module load anaconda/2023a

# Activate the Conda environment
source activate myenv

# Print debug information
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "SLURM_NNODES = $SLURM_NNODES"
echo "SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE"

# Run your Python script
python test_script.py

# Print debug information
echo "Job finished at $(date)"
