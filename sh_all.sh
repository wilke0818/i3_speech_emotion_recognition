#!/bin/bash                      
#SBATCH --job-name=ser
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err
#SBATCH -t 96:00:00          # walltime = 1 hours and 30 minutes
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fabiocat@mit.edu
# Set the array variable based on the calculated array size
#SBATCH -N 1
#SBATCH -n 8                     # 1 CPU core
#SBATCH --mem=240GB
#SBATCH --gres=shard:1 
#SBATCH -x node[100-106,110]
#SBATCH --constraint=any-A100
#SBATCH -p gablab
# Execute commands to run your program here. Here is an example of python.
eval "$(conda shell.bash hook)"
conda activate ser

# Print the current task information
echo "Running run.py"
python run.py experiments/batch_and_dropout.json
