#!/bin/bash

echo " "  # Print a blank line for spacing

# Initialize arrays to store test results for original and Voronoi patch types
original_results=()
voronoi_results=()

# Function to run patch tests for a given patch type and store accuracies in an array
run_patch_tests() {
    patch_type=$1             # Patch type: e.g., "tumor" or "voronoi"
    results_lst_name=$2       # Name of the array variable to store results
    num_simulations=$3        # Number of test iterations to run

    count=0
    while [ $count -lt $num_simulations ]; do
        # Run the Python script with the specified patch type and capture its output
        accuracy=$(python ResNetModel.py --type "$patch_type" | tail -n 1)

        # Check if the output is not empty
        if [[ -n "$accuracy" ]]; then
            # Append the accuracy output to the array whose name is stored in results_lst_name
            eval "$results_lst_name+=(\"\$accuracy\")"
            ((count++))
        else
            echo "Test did not pass or returned empty output; retrying..."
        fi
    done
}

# Set the number of runs per patch type
num_simulations=1

# Run tests for both tumor and voronoi patches, storing results in respective arrays
run_patch_tests "tumor" original_results $num_simulations
run_patch_tests "voronoi" voronoi_results $num_simulations

# Convert Bash arrays to JSON strings using jq for passing as arguments to the Python script
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
