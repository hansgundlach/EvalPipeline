#!/bin/bash
#SBATCH --job-name=my_python_job  # Job name
#SBATCH --output=output_%j.txt    # Output file (%j will be replaced with the job ID)
#SBATCH --error=error_%j.txt      # Error file (%j will be replaced with the job ID)
#SBATCH --time=01:00:00           # Time limit (hh:mm:ss)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem=4GB                 # Memory per node

# Load necessary modules
module load python/3.8  # Adjust based on your required Python version

# Activate your virtual environment if needed
source ~/myenv/bin/activate  # Adjust based on your virtual environment path

# Run your Python script
python my_script.py
