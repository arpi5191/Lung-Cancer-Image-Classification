#!/bin/bash
#SBATCH -N 1                                  # Request 1 node
#SBATCH --job-name=resnet_test                # Job name
#SBATCH --output=resnet_out.txt               # File to save standard output
#SBATCH --error=resnet_err.txt                # File to save standard error
#SBATCH --gres=gpu:4                          # Request 4 GPUs for parallel runs
#SBATCH --time=05:00:00                       # Max runtime (30 minutes); adjust as needed
#SBATCH --mem=32G                              # CPU memory for data loading/preprocessing
#SBATCH --partition=GPU-shared                # GPU partition
#SBATCH --mail-type=END                       # Send email when job ends
#SBATCH --mail-user=arpi5191@gmail.com        # Your email address

# Activate your conda environment
# source /ocean/projects/bio240001p/arpitha/myenv/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

echo " "  # Print a blank line for spacing

# Initialize arrays to store test results for original and Voronoi patch types
original_results=()
voronoi_results=()

# Function to run patch tests for a given patch type and store accuracies in an array
run_patch_tests() {
    patch_type=$1             # Patch type: "tumor" or "voronoi"
    results_lst_name=$2       # Name of the array variable to store results
    num_simulations=$3        # Total number of runs to perform

    count=0                   # Counter for total completed simulations

    # Loop until we have collected the desired number of simulation results
    while [ $count -lt $num_simulations ]; do

        # -----------------------------
        # Run a single Python simulation
        # -----------------------------
        # - Use all 4 GPUs for this single run by specifying them in CUDA_VISIBLE_DEVICES
        # - -u option ensures unbuffered output, so we can read it in real-time
        # - Pipe output to grep to extract numeric values (accuracy)
        # - Use tail -n 1 to pick the last numeric value if multiple lines are printed
        # Credit to chatGPT
        accuracy=$(CUDA_VISIBLE_DEVICES=0,1,2,3 python -u ResNetModel.py --type "$patch_type" \
                    | grep -Eo '^[0-9]*\.?[0-9]+$' \
                    | tail -n 1)

        # -----------------------------
        # Check if the output is valid
        # -----------------------------
        if [[ -n "$accuracy" ]]; then
            # Append the accuracy output to the array whose name is stored in results_lst_name
            eval "$results_lst_name+=(\"\$accuracy\")"
            ((count++))   # Increment the counter for total completed simulations
        else
            echo "Simulation did not return a valid accuracy; retrying..."
        fi

    done
}

# Set the number of runs per patch type
num_simulations=20

# Run tests for both tumor and voronoi patches, storing results in respective arrays
run_patch_tests "tumor" original_results $num_simulations
run_patch_tests "voronoi" voronoi_results $num_simulations

# Convert Bash arrays to JSON strings using jq for passing as arguments to the Python script
# Credit to chatGPT
original_json=$(printf '%s\n' "${original_results[@]}" | jq -R . | jq -s .)
voronoi_json=$(printf '%s\n' "${voronoi_results[@]}" | jq -R . | jq -s .)

# Print JSON strings for verification/debugging
echo " "
echo "Original JSON:"
echo "$original_json"
echo " "
echo "Voronoi JSON:"
echo "$voronoi_json"
echo " "

# Run the Wilcoxon test Python script with the JSON-formatted arrays as input
# Output (p-value) will be printed directly by the Python script
echo "Running Wilcoxon test..."
python wilcoxon_test.py --original "$original_json" --voronoi "$voronoi_json"
echo "Finished Wilcoxon test."
echo " "

# Time for 1 simulation on Bash script w/2 epochs per input type: 16 mins 25 secs

# Time for 1 simulation on Bash script w/1 epoch per input type: 8 mins
# Time for 1 simulation on Bash script w/30 epochs per input type: 4 hrs
# Time for 20 simulations on Bash script w/30 epochs per input type: 80 hrs
