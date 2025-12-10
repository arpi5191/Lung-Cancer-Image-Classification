import json
import argparse
import numpy as np
from scipy.stats import wilcoxon

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Perform Wilcoxon signed-rank test between two patch datasets.")
    parser.add_argument('--type1', required=True, help='JSON list of accuracies from patch type 1')
    parser.add_argument('--type2', required=True, help='JSON list of accuracies from patch type 2')
    args = parser.parse_args()

    # Convert JSON strings to Python lists of floats
    data1 = list(map(float, json.loads(args.type1)))
    data2 = list(map(float, json.loads(args.type2)))

    # Compute and print medians
    median1 = np.median(data1)
    median2 = np.median(data2)
    print(f"Type 1 median accuracy: {median1}")
    print(f"Type 2 median accuracy: {median2}")

    # Perform Wilcoxon signed-rank test
    stat, p_value = wilcoxon(data1, data2)

    # Print significance
    if p_value < 0.05:
        print("Statistically significant difference (p < 0.05).")
    else:
        print("No statistically significant difference (p >= 0.05).")

if __name__ == "__main__":
    main()
