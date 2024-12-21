#!/bin/bash
#SBATCH --partition=gpu-single       # GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --time=08:00:00              # Max runtime
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=40gb                   # Memory allocation
#SBATCH --job-name=propensity_job    # Job name
#SBATCH --output=run_%j.out          # Standard output file
#SBATCH --error=run_%j.err           # Standard error file

# Load the CUDA module (replace with your CUDA version)
module load devel/cuda/12.6

# Activate the Python virtual environment
source /gpfs/bwfor/home/tu/tu_tu/tu_wwdth01/DPN-SA/Jobs/venv/bin/activate

# Move to the directory containing your script
cd /gpfs/bwfor/home/tu/tu_tu/tu_wwdth01/DPN-SA/Jobs

# Optional: Log GPU and CUDA information
echo "Available GPUs:"
nvidia-smi
echo "CUDA version:"
nvcc --version

# Verify PyTorch and CUDA compatibility
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('CUDNN version:', torch.backends.cudnn.version())
    print('Device name:', torch.cuda.get_device_name(0))
else:
    raise RuntimeError('CUDA is not available. Check your setup.')
"


