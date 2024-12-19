#!/bin/bash
#SBATCH --partition=gpu-single       # GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --time=00:10:00              # Max runtime
#SBATCH --mem=4gb                    # Memory
#SBATCH --job-name=check_cuda_torch  # Job name
#SBATCH --output=check_cuda_torch_%j.out  # Standard output file
#SBATCH --error=check_cuda_torch_%j.err   # Standard error file

# Load the correct CUDA module (11.6)
module load devel/cuda/12.6

# Activate your new virtual environment
source /gpfs/bwfor/home/tu/tu_tu/tu_wwdth01/DPN-SA/Jobs/venv/bin/activate

# Check CUDA and PyTorch compatibility
python - << EOF
import torch
print("Checking PyTorch and CUDA configuration...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected or CUDA is not available.")
EOF
