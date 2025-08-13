# Import necessary packages
import json
import argparse
from scipy.stats import wilcoxon

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser()

    # Define required command-line arguments:
    # --original: JSON string list of accuracies from original patch tests
    # --voronoi: JSON string list of accuracies from Voronoi patch tests
    parser.add_argument('--original', required=True, help='JSON list of accuracies from original patch')
    parser.add_argument('--voronoi', required=True, help='JSON list of accuracies from Voronoi patch')

    # Parse arguments
    args = parser.parse_args()

    # Convert JSON strings to Python lists of floats for statistical testing
    original = list(map(float, json.loads(args.original)))
    voronoi = list(map(float, json.loads(args.voronoi)))

    # Perform Wilcoxon signed-rank test to compare paired samples
    stat, p_value = wilcoxon(original, voronoi)

    # Print significance result based on p-value threshold 0.05
    if p_value < 0.05:
        print("Statistically significant (p < 0.05).")
    else:
        print("Not statistically significant (p >= 0.05).")

if __name__ == "__main__":
    main()
