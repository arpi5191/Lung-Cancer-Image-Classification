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
