#!/bin/bash
#SBATCH --partition=gpu-single       # GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --time=01:00:00              # Max runtime
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=4gb                    # Memory
#SBATCH --job-name=propensity_job    # Job name
#SBATCH --output=run_%j.out   # Standard output file (%j will be replaced with the job ID)
#SBATCH --error=run_%j.err    # Standard error file

# Load the correct CUDA module (12.1)
module load devel/cuda/12.1

# Activate the Python virtual environment
source /home/tu/tu_tu/tu_wwdth01/DPN-SA-MMD/Jobs/venv/bin/activate

# Move to the directory containing your script
cd /home/tu/tu_tu/tu_wwdth01/DPN-SA-MMD/Jobs/

python main_mmd.py


