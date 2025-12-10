#!/bin/bash

# ----------------------------
# Initialize arrays to store accuracy results for each patch type
# ----------------------------
original_results=()      # Accuracies from the original (tumor) patches
voronoi_results=()      # Accuracies from Voronoi patches
diffusion_results=()    # Accuracies from diffusion patches
prompt_results=()       # Accuracies from prompt-generated patches
context_results=()      # Accuracies from context-generated patches

# ----------------------------
# Function to run patch tests multiple times for a given patch type
# Arguments:
#   1: patch_type       (string) - type of patch: tumor, voronoi, diffusion, prompt, context
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

        # Run ResNetModel.py and extract the last numeric output as accuracy
        accuracy=$(python -u ResNetModel.py --type "$patch_type" 2>&1 \
                    | tee -a resnet_out.txt \
                    | grep -Eo '^[0-9]*\.?[0-9]+$' \
                    | tail -n 1)

        # Check if the output is a valid number
        if [[ -n "$accuracy" && "$accuracy" =~ ^[0-9]*\.?[0-9]+$ ]]; then
            # Append the accuracy to the corresponding results array
            eval "$results_lst_name+=(\"\$accuracy\")"
            echo "Simulation $((count + 1)) completed. Accuracy: $accuracy"
            ((count++))   # Increment the counter
        else
            echo "Simulation did not return a valid accuracy; retrying..."
        fi
    done

    echo "Completed all $num_simulations simulations for $patch_type patches"
    echo " "
}

# ----------------------------
# Number of simulation runs per patch type
# ----------------------------
num_simulations=20

# ----------------------------
# Run patch tests for each type and store results in corresponding arrays
# ----------------------------
run_patch_tests "tumor" original_results $num_simulations
run_patch_tests "voronoi" voronoi_results $num_simulations
run_patch_tests "diffusion" diffusion_results $num_simulations
run_patch_tests "prompt" prompt_results $num_simulations
run_patch_tests "context" context_results $num_simulations

# ----------------------------
# Convert Bash arrays to JSON format for passing to Python scripts
# ----------------------------
original_json=$(printf '%s\n' "${original_results[@]}" | jq -R . | jq -s .)
voronoi_json=$(printf '%s\n' "${voronoi_results[@]}" | jq -R . | jq -s .)
diffusion_json=$(printf '%s\n' "${diffusion_results[@]}" | jq -R . | jq -s .)
prompt_json=$(printf '%s\n' "${prompt_results[@]}" | jq -R . | jq -s .)
context_json=$(printf '%s\n' "${context_results[@]}" | jq -R . | jq -s .)

# Print JSON for verification/debugging
echo "Original JSON: $original_json"
echo "Voronoi JSON: $voronoi_json"
echo "Diffusion JSON: $diffusion_json"
echo "Prompt JSON: $prompt_json"
echo "Context JSON: $context_json"
echo " "

# ----------------------------
# Define which pairs of datasets to compare with Wilcoxon test
# ----------------------------
comparisons=(
    "original voronoi"
    "original diffusion"
    "original prompt"
    "original context"
    "prompt diffusion"
    "prompt context"
    "diffusion context"
)

# ----------------------------
# Loop over each pair and run Wilcoxon signed-rank test
# ----------------------------
for pair in "${comparisons[@]}"; do
    set -- $pair
    type1=$1
    type2=$2
    echo "Running Wilcoxon test for $type1 vs $type2..."

    # Call the Python script and pass the JSON arrays for the two datasets
    python wilcoxon_test.py --type1 "$(eval echo \${${type1}_json})" \
                            --type2 "$(eval echo \${${type2}_json})"

    echo "Finished Wilcoxon test for $type1 vs $type2."
    echo ""
done
