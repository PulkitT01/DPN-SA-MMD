#!/bin/bash
#SBATCH --partition=gpu-single       # GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --time=00:05:00              # Max runtime
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=40gb                   # Memory allocation
#SBATCH --job-name=propensity_job    # Job name
#SBATCH --output=test_%j.out          # Standard output file
#SBATCH --error=test_%j.err           # Standard error file

# Load the CUDA module (replace with your CUDA version)
module load devel/cuda/12.1

# Activate the Python virtual environment
source /home/tu/tu_tu/tu_wwdth01/DPN-SA-MMD/Jobs/venv/bin/activate

# Move to the directory containing your script
cd /home/tu/tu_tu/tu_wwdth01/DPN-SA-MMD/Jobs/

# Optional: Log GPU and CUDA information
echo "Available GPUs:"
nvidia-smi
echo "CUDA version:"
nvcc --version

# Verify PyTorch and CUDA compatibility
# Verify Python version, PyTorch, and CUDA compatibility
python -c "
import sys
import torch

# Check Python version
print(f'Python executable: {sys.executable}')
print(f'Python version: {sys.version}')

# Verify PyTorch installation
try:
    print(f'PyTorch version: {torch.__version__}')
except ImportError as e:
    raise RuntimeError('PyTorch is not installed. Please install PyTorch.') from e

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f'CUDA available: {cuda_available}')
if cuda_available:
    print(f'CUDA version: {torch.version.cuda}')
    print(f'CUDNN version: {torch.backends.cudnn.version()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Number of CUDA devices: {torch.cuda.device_count()}')
else:
    print('CUDA is not available. Check your setup.')
    raise RuntimeError('CUDA is not available.')

# Check if PyTorch and CUDA are compatible
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    x = torch.tensor([1.0], device=device)
    print(f'Tensor successfully created on {device}: {x}')
except Exception as e:
    raise RuntimeError('Error initializing CUDA with PyTorch. Check your installation.') from e
"



