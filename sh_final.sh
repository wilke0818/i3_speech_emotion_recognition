#!/bin/bash                      
#SBATCH --job-name=ser
#SBATCH --output=./logs_final/%A.out
#SBATCH --error=./logs_final/%A.err
#SBATCH -t 8:00:00          # walltime = 1 hours and 30 minutes
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wilke18@mit.edu
# Set the array variable based on the calculated array size
#SBATCH -N 1
#SBATCH -n 8                     # 1 CPU core
#SBATCH --mem=40GB
#SBATCH --gres=gpu:a100:1 
#SBATCH -x node[100-106,110]

# Execute commands to run your program here. Here is an example of python.
eval "$(conda shell.bash hook)"
conda activate ser

let "min_seed = 14"
let "max_seed = 15"

# Print the current task information
echo "Running train.py with min_seed = $min_seed and max_seed = $max_seed"
python train.py --experiment_file experiments/run.json --low_seed "$min_seed" --high_seed "$max_seed"
