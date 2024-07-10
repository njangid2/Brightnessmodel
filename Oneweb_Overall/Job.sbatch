#!/bin/bash
#####################################################
## Important parameters of job ##
## Setup SLURM submission options ##
#####################################################
#SBATCH --time=9:00:10
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output="LOG.log"
#SBATCH --job-name="myjobname"
#SBATCH --partition=caps
#SBATCH --mail-type=ALL
#SBATCH --mail-user="njangid2@illinois.edu"

########## Debugging and environment setup ######################
echo "Running job on host: $(hostname)"
echo "Starting at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node list: $SLURM_NODELIST"

## Load required modules (Step 3 has been done)
module load anaconda/3
source activate my.anaconda

## Confirm environment activation
echo "Activated Conda Environment: $(conda info --envs | grep '*')"

## Check the python version and environment
which python
python --version

## Run script
python3 /home/njangid2/final/onee.py

## End of job
echo "Finished at: $(date)"
