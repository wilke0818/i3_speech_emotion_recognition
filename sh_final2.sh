#!/bin/bash                      
#SBATCH --job-name=ser
#SBATCH --output=./logs_final/%A_%a.out
#SBATCH --error=./logs_final/%A_%a.err
#SBATCH -t 150:00:00          # walltime = 1 hours and 30 minutes
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fabiocat@mit.edu
# Set the array variable based on the calculated array size
#SBATCH -N 1
#SBATCH -n 8                     # 1 CPU core
#SBATCH --mem=240GB
#SBATCH --gres=gpu:a100:1 
#SBATCH -x node[100-106,110]
#SBATCH -p gablab
#SBATCH --array=7-11

# Execute commands to run your program here. Here is an example of python.
eval "$(conda shell.bash hook)"
conda activate ser

let "min_seed = ($SLURM_ARRAY_TASK_ID - 1) * 10"
let "max_seed = ($SLURM_ARRAY_TASK_ID) * 10"

# Print the current task information
echo "Running run.py with min_seed = $min_seed and max_seed = $max_seed"
python run.py --experiment_file experiments_final/run.json --low_seed "$min_seed" --high_seed "$max_seed"
