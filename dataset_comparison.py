import json
import argparse
import numpy as np
from scipy.stats import wilcoxon

def main():
    # -----------------------------
    # Parse command-line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Perform Wilcoxon signed-rank test and compute bootstrap CI for two patch datasets."
    )
    parser.add_argument('--type1', required=True, help='JSON list of F1 scores from patch type 1')
    parser.add_argument('--type2', required=True, help='JSON list of F1 scores from patch type 2')
    args = parser.parse_args()

    # -----------------------------
    # Convert JSON strings to Python lists of floats
    # -----------------------------
    f1_scores_1 = list(map(float, json.loads(args.type1)))
    f1_scores_2 = list(map(float, json.loads(args.type2)))

    # -----------------------------
    # Compute and print medians
    # -----------------------------
    median1 = np.median(f1_scores_1)
    median2 = np.median(f1_scores_2)
    print(f"Type 1 median F1 score: {median1:.4f}")
    print(f"Type 2 median F1 score: {median2:.4f}")
    print()  # blank line for readability

    # -----------------------------
    # Compute and print means
    # -----------------------------
    mean1 = np.mean(f1_scores_1)
    mean2 = np.mean(f1_scores_2)
    print(f"Type 1 mean F1 score: {mean1:.4f}")
    print(f"Type 2 mean F1 score: {mean2:.4f}")
    print()

    # -----------------------------
    # Compute 95% bootstrap confidence intervals for median
    # -----------------------------
    n_bootstrap = 1000  # number of bootstrap samples
    boot_medians_1 = []
    boot_medians_2 = []

    # Generate bootstrap resamples for dataset 1
    for _ in range(n_bootstrap):
        sample = np.random.choice(f1_scores_1, size=len(f1_scores_1), replace=True)
        boot_medians_1.append(np.median(sample))

    # Generate bootstrap resamples for dataset 2
    for _ in range(n_bootstrap):
        sample = np.random.choice(f1_scores_2, size=len(f1_scores_2), replace=True)
        boot_medians_2.append(np.median(sample))

    # Compute 95% confidence intervals
    ci1_lower, ci1_upper = np.percentile(boot_medians_1, [2.5, 97.5])
    ci2_lower, ci2_upper = np.percentile(boot_medians_2, [2.5, 97.5])

    print(f"Type 1 95% bootstrap CI: [{ci1_lower:.4f}, {ci1_upper:.4f}]")
    print(f"Type 2 95% bootstrap CI: [{ci2_lower:.4f}, {ci2_upper:.4f}]")
    print()

    # -----------------------------
    # Perform Wilcoxon signed-rank test (paired samples)
    # -----------------------------
    stat, p_value = wilcoxon(f1_scores_1, f1_scores_2)

    # Print statistical significance
    if p_value < 0.05:
        print(f"Statistically significant difference (p = {p_value:.4f} < 0.05).")
    else:
        print(f"No statistically significant difference (p = {p_value:.4f} >= 0.05).")

    # -----------------------------
    # Compute rank-biserial correlation (effect size) for paired differences
    # -----------------------------
    diffs = np.array(f1_scores_1) - np.array(f1_scores_2)  # ensure numpy arrays
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    n_zero = np.sum(diffs == 0)
    rbs = (n_pos - n_neg) / (n_pos + n_neg + n_zero)

    # Print the effect size
    print(f"Rank-biserial correlation (effect size): {rbs:.3f}")

if __name__ == "__main__":
    main()
