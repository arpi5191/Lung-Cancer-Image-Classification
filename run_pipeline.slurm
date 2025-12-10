#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=resnet_test
#SBATCH --output=resnet_out.txt
#SBATCH --error=resnet_err.txt
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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
mkdir -p $CONDA_PKGS_DIRS $CONDA_ENVS_DIRS

# ----------------------------
# Clean caches
# ----------------------------
echo "Cleaning conda and pip cache..."
conda clean --all -y
pip cache purge

# ----------------------------
# Create fresh Conda environment
# ----------------------------
echo "Creating new conda environment..."
conda create -p /ocean/projects/bio240001p/arpitha/conda_envs/myenv_final python=3.10 -y
conda activate /ocean/projects/bio240001p/arpitha/conda_envs/myenv_temp

# ----------------------------
# Install packages
# ----------------------------
echo "Installing packages from requirements.txt..."
pip install --no-cache-dir --timeout=600 -r requirements.txt

# ----------------------------
# Set GPU visibility
# ----------------------------
export CUDA_VISIBLE_DEVICES=0
nvidia-smi

# ----------------------------
# Run segmentation scripts
# ----------------------------
# python tumor_seg.py --didx 0 --d 2
# python voronoi_seg.py
python diffusion_seg.py
python prompt_seg.py
python context_seg.py

# ----------------------------
# Extract patches
# ----------------------------
python extract_patches.py --type tumor
python extract_patches.py --type voronoi
python extract_patches.py --type diffusion
python extract_patches.py --type prompt
python extract_patches.py --type context

# ----------------------------
# Make the compare_models script executable and run the ResNet simulations
# This will run all patch tests and perform Wilcoxon statistical comparisons
# ----------------------------
chmod +x compare_models.sh
./compare_models.sh

echo ""

echo "All steps completed successfully!"

# ----------------------------
# Deactivate environment
# ----------------------------
conda deactivate

echo "Job complete. Environment preserved for next script."
