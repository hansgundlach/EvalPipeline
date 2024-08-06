#!/bin/bash
# set -x
# set -e
rm -f "$log_directory"/*
#SBATCH -o job_logs/log-%j
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8

# Loading the required module

module load anaconda/2023a
module load cuda/11.8
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a-pytorch/etc/profile.d/conda.sh

conda init bash

HF_USER_DIR="/home/gridsan/$(whoami)/"
# HF_USER_DIR="/home/gridsan/$(whoami)/futuretech_shared/atrisovic/osfm/paper_analysis_toolkit/"
HF_LOCAL_DIR="/state/partition1/user/$(whoami)/cache/huggingface"
# mkdir -p $HF_LOCAL_DIR
# rsync -a --ignore-existing $HF_USER_DIR/ ${HF_LOCAL_DIR}
export HF_HOME=${HF_LOCAL_DIR}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED="true"

job_number=$1
total_jobs=$2

# Check available environments
echo "Available conda environments:"
conda info --envs

echo "Setting up Environment"
# conda update tensorflow
conda activate py310


echo "I'm navigating to the directory {/home/gridsan/$(whoami)/EvalPipeline}"
# Navigate to the directory containing your scripts and models
cd /home/gridsan/$(whoami)/EvalPipeline

#python affiliations_main.py -d -f data/temp_open_access_paper_ids.csv -i $job_number -n $total_jobs
#python citations_main.py -d -f data/open_access_paper_ids.csv -i $job_number -n $total_jobs
#python citations_main.py -d -i $job_number -n $total_jobs
# python question_set.py:Q

python gpt2_from_load.py
