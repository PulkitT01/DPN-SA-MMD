#!/bin/bash
#SBATCH --partition=gpu-single       # GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --time=02:00:00              # Max runtime
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=40gb                   # Memory
#SBATCH --job-name=propensity_job    # Job name
#SBATCH --output=run_%j.out   # Standard output file (%j will be replaced with the job ID)
#SBATCH --error=run_%j.err    # Standard error file

# Load the correct CUDA module (12.1)
module load devel/cuda/12.1

# Activate your new virtual environment
source /gpfs/bwfor/home/tu/tu_tu/tu_wwdth01/DPN-SA/Jobs/venv/bin/activate

cd /gpfs/bwfor/home/tu/tu_tu/tu_wwdth01/DPN-SA/Jobs

python main_propensity_dropout.py


