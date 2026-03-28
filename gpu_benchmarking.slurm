#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=resnet_gpu_test
#SBATCH --output=resnet_gpu_out.txt
#SBATCH --error=resnet_gpu_err.txt
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --partition=GPU-shared
#SBATCH --mail-type=END
#SBATCH --mail-user=arpi5191@gmail.com

set -e

# ----------------------------
# Load necessary modules
# ----------------------------
module purge
module load anaconda3/2024.10-1
module load cuda/12.6.1

# ----------------------------
# Initialize Conda
# ----------------------------
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh

# ----------------------------
# Force Conda to use project space
# ----------------------------
export CONDA_PKGS_DIRS=/ocean/projects/bio240001p/arpitha/conda_pkgs
export CONDA_ENVS_DIRS=/ocean/projects/bio240001p/arpitha/conda_envs
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS"

# ----------------------------
# Create fresh Conda environment (optional)
# ----------------------------
# Uncomment if you want to create a new environment each run
# echo "Creating new conda environment..."
# conda create -p /ocean/projects/bio240001p/arpitha/conda_envs/myenv python=3.10 -y

# ----------------------------
# Activate Conda environment
# ----------------------------
conda activate /ocean/projects/bio240001p/arpitha/conda_envs/myenv

# ----------------------------
# Install packages (optional)
# ----------------------------
# Uncomment if you need to install packages each run
# echo "Installing packages from requirements.txt..."
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# pip install --no-cache-dir --timeout=600 -r requirements.txt

# ----------------------------
# Prepare GPU metrics folder
# ----------------------------
FOLDER="results/gpu_metrics"

rm -rf gpu_metrics && mkdir -p gpu_metrics
echo "Folder reset: results/gpu_metrics"

# -----------------------------
# Run the ResNetModel.py script with different GPU workloads
# Loops over 1 to 4 GPUs and sets CUDA_VISIBLE_DEVICES
# -----------------------------
for NGPU in 1 2 3 4; do
    # Create a comma-separated list of GPU indices for this run
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU - 1)))
    export CUDA_VISIBLE_DEVICES

    # Print which GPUs are being used
    echo "Running with $NGPU GPU(s): $CUDA_VISIBLE_DEVICES"

    # Run the training script with the specified GPU setup
    python ResNetModel.py --type context
done

# ----------------------------
# Graph GPU metrics
# ----------------------------
# Run the plotting script
python gpu_plots.py

# ----------------------------
# Deactivate environment
# ----------------------------
conda deactivate
echo "Job complete. Environment preserved for next script."
