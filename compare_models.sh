#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=resnet_test
#SBATCH --output=resnet_out.txt
#SBATCH --error=resnet_err.txt
#SBATCH --gres=gpu:3
#SBATCH --time=07:00:00
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
conda create -p /ocean/projects/bio240001p/arpitha/conda_envs/myenv python=3.10 -y
conda activate /ocean/projects/bio240001p/arpitha/conda_envs/myenv

# ----------------------------
# Install packages
# ----------------------------
echo "Installing packages from requirements.txt..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install --no-cache-dir --timeout=600 -r requirements.txt

# ----------------------------
# Set GPU visibility
# ----------------------------
export CUDA_VISIBLE_DEVICES=0,1,2
# export CUDA_VISIBLE_DEVICES=0
nvidia-smi

# ----------------------------
# Run segmentation scripts
# ----------------------------
python tumor_seg.py --didx 0 --d 2
python voronoi_seg.py
python context_seg.py

# ----------------------------
# Extract patches
# ----------------------------
python extract_patches.py --type tumor
python extract_patches.py --type voronoi
python extract_patches.py --type context

# ----------------------------
# Make the compare_models script executable and run the ResNet simulations
# This will run all patch tests and perform Wilcoxon statistical comparisons
# ----------------------------
chmod +x compare_models.sh
./compare_models.sh

#----------------------------
# # Print completion message after all pipeline steps finish successfully
#----------------------------
echo ""
echo "All steps completed successfully!"

# ----------------------------
# Deactivate environment
# ----------------------------
conda deactivate
echo "Job complete. Environment preserved for next script."
[arpitha@bridges2-login011 arpitha]$ cat compare_models.sh
#!/bin/bash

# ----------------------------
# Initialize arrays to store accuracy results for each patch type
# ----------------------------
original_results=()      # Accuracies from the original (tumor) patches
voronoi_results=()      # Accuracies from Voronoi patches
synthetic_results=()      # Accuracies from context-generated patches

# ----------------------------
# Function to run patch tests multiple times for a given patch type
# Arguments:
#   1: patch_type       (string) - type of patch: tumor, voronoi, context
#   2: results_lst_name (string) - name of the array to store accuracies
#   3: num_simulations  (int)    - number of runs to perform
# ----------------------------
run_patch_tests() {
    patch_type=$1
    results_lst_name=$2
    num_simulations=$3

    count=0
    echo "Starting $num_simulations simulations for patch type: $patch_type"

    # Loop until the desired number of simulations is completed
    while [ $count -lt $num_simulations ]; do
        echo "Running simulation $((count + 1))/$num_simulations for $patch_type patches..."

        # Run ResNetModel.py and extract the last numeric output as F1 score
        f1_score=$(python -u ResNetModel.py --type "$patch_type" 2>&1 \
                    | tee -a resnet_out.txt \
                    | grep -Eo '^[0-9]*\.?[0-9]+$' \
                    | tail -n 1)

        # Check if the output is a valid number
        if [[ -n "$f1_score" && "$f1_score" =~ ^[0-9]*\.?[0-9]+$ ]]; then
            # Append the F1 score to the corresponding results array
            eval "$results_lst_name+=(\"\$f1_score\")"
            echo "Simulation $((count + 1)) completed. F1 Score: $f1_score"
            ((count++))   # Increment the counter
        else
            echo "Simulation did not return a valid F1 score; retrying..."
        fi
        done

    echo "Completed all $num_simulations simulations for $patch_type patches"
    echo " "
}

# ----------------------------
# Number of simulation runs per patch type
# ----------------------------
num_simulations=50

# ----------------------------
# Run patch tests for each type and store results in corresponding arrays
# ----------------------------
run_patch_tests "tumor" original_results $num_simulations
run_patch_tests "voronoi" voronoi_results $num_simulations
run_patch_tests "synthetic" synthetic_results $num_simulations

# ----------------------------
# Convert Bash arrays to JSON format for passing to Python scripts
# ----------------------------
original_json=$(printf '%s\n' "${original_results[@]}" | jq -R . | jq -s .)
voronoi_json=$(printf '%s\n' "${voronoi_results[@]}" | jq -R . | jq -s .)
synthetic_json=$(printf '%s\n' "${synthetic_results[@]}" | jq -R . | jq -s .)

# ----------------------------
# Print JSON for verification/debugging
# ----------------------------
echo "Original JSON: $original_json"
echo "Voronoi JSON: $voronoi_json"
echo "Synthetic JSON: $synthetic_json"
echo " "
echo " "

# ----------------------------
# Run the Python script with the JSON arrays passed as command-line arguments
# ----------------------------
python dataset_comparison.py --regular "$original_json" --voronoi "$voronoi_json" --synthetic "$synthetic_json"
