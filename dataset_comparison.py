# Imports packages
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

def find_means(label, f1_scores):
    """
    Compute and print the mean F1 score.

    Parameters
    ----------
    label : str
        A descriptive name for the dataset or model (e.g., "Model A").
    f1_scores : array-like
        A list or NumPy array of F1 scores.

    Returns
    -------
    float
        The mean F1 score.
    """
    # Compute mean of the F1 scores
    mean_score = np.mean(f1_scores)

    # Print formatted result
    print(f"{label} mean F1 score: {mean_score:.4f}")

    # Add a line for readability
    print()

def find_medians(label, f1_scores):
    """
    Compute and print the median F1 score.

    Parameters
    ----------
    label : str
        A descriptive name for the dataset or model (e.g., "Model A").
    f1_scores : array-like
        A list or NumPy array of F1 scores.

    Returns
    -------
    float
        The median F1 score.
    """
    # Compute the median of the F1 scores
    median_score = np.median(f1_scores)

    # Print formatted result
    print(f"{label} median F1 score: {median_score:.4f}")

    # Add a line for readability
    print()

def bootstrap_medians(label, f1_scores, n_bootstrap=1000):
    """
    Generate a bootstrap distribution of medians from F1 scores and compute a 95% confidence interval.

    Bootstrapping involves sampling with replacement from the input data multiple times,
    computing the median for each resample. This can be used to estimate confidence intervals
    or the variability of the median.

    Parameters
    ----------
    label : str
        A descriptive name for the dataset or model (e.g., "Model A") used for printing.
    f1_scores : array-like
        A list or NumPy array of F1 scores.
    n_bootstrap : int, optional (default=1000)
        The number of bootstrap resamples to generate.

    Returns
    -------
    tuple
        - np.ndarray: Array of bootstrapped medians.
        - float: Lower bound of the 95% confidence interval.
        - float: Upper bound of the 95% confidence interval.
    """
    # List to store median of each bootstrap sample
    boot_medians = []

    # Perform bootstrap resampling
    for _ in range(n_bootstrap):
        # Sample with replacement from the regular F1 scores
        sample = np.random.choice(f1_scores, size=len(f1_scores), replace=True)
        # Compute and store the median of the sample
        boot_medians.append(np.median(sample))

    # Convert list to NumPy array for easier downstream analysis
    boot_medians = np.array(boot_medians)

    # Compute the median of the bootstrapped medians (point estimate of central tendency)
    boot_median = np.median(boot_medians)

    # Compute 95% confidence interval from the bootstrapped medians
    ci_lower, ci_upper = np.percentile(boot_medians, [2.5, 97.5])

    # Print the bootstrap median and confidence interval
    print(f"{label} bootstrap median: {boot_median:.4f}, "
          f"95% bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Add a line for readability
    print()

    return boot_medians, boot_median, ci_lower, ci_upper

def plot_all_bootstraps(regular_boot, regular_ci, regular_median,
                        voronoi_boot, voronoi_ci, voronoi_median,
                        synthetic_boot, synthetic_ci, synthetic_median,
                        save_path):
    """
    Plot bootstrap histograms for Regular, Voronoi, and Synthetic datasets
    in a single figure and save to disk.

    Parameters
    ----------
    regular_boot, voronoi_boot, synthetic_boot : array-like
        Bootstrapped median values for each dataset.
    regular_ci, voronoi_ci, synthetic_ci : tuple(float, float)
        Confidence intervals (lower, upper) for each dataset.
    regular_median, voronoi_median, synthetic_median : float
        Precomputed median values for each dataset.
    save_path : str
        File path where the figure will be saved (e.g., "results/all_bootstrap.png").

    Returns
    -------
    None
        Saves the figure to disk.
    """

    # Bundle dataset info for cleaner iteration
    datasets = [
        ("Regular", regular_boot, regular_ci, regular_median),
        ("Voronoi", voronoi_boot, voronoi_ci, voronoi_median),
        ("Synthetic", synthetic_boot, synthetic_ci, synthetic_median)
    ]

    # Create 3 side-by-side subplots (shared y-axis for easier comparison)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (label, boot_vals, ci, median) in zip(axes, datasets):
        ci_lower, ci_upper = ci

        # Plot histogram of bootstrap distribution
        ax.hist(boot_vals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

        # Plot confidence interval bounds (dashed red lines)
        ax.axvline(ci_lower, color='red', linestyle='--',
                   label=f"CI lower ({ci_lower:.4f})")
        ax.axvline(ci_upper, color='red', linestyle='--',
                   label=f"CI upper ({ci_upper:.4f})")

        # Plot median (solid green line)
        ax.axvline(median, color='green', linestyle='-',
                   label=f"Median ({median:.4f})")

        # Axis labels and title
        ax.set_title(label)
        ax.set_xlabel("Bootstrapped F1 Values")
        ax.set_ylabel("Frequency")

        # Show legend for each subplot
        ax.legend()

    # Adjust layout to prevent overlap between subplots
    plt.tight_layout()

    # Save figure to disk
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"All bootstrap histograms saved in one figure: {save_path}")

def plot_all_error_plots(regular_ci, regular_median,
                         voronoi_ci, voronoi_median,
                         synthetic_ci, synthetic_median,
                         save_path):
    """
    Plot median F1 scores with 95% confidence intervals for
    Regular, Voronoi, and Synthetic datasets using asymmetric error bars,
    with clear labeling and a legend.

    Parameters
    ----------
    regular_ci, voronoi_ci, synthetic_ci : tuple(float, float)
        Confidence intervals (lower, upper) for each dataset.
    regular_median, voronoi_median, synthetic_median : float
        Median F1 scores for each dataset.
    save_path : str
        File path to save the error bar plot (e.g., "results/error_plot.png").

    Returns
    -------
    None
        Saves the error bar plot to disk.
    """

    # Bundle dataset info into a list for easy iteration
    # Each tuple: (label, confidence interval, median, color, marker style)
    datasets = [
        ("Regular", regular_ci, regular_median, 'blue', 'o'),
        ("Voronoi", voronoi_ci, voronoi_median, 'green', 's'),
        ("Synthetic", synthetic_ci, synthetic_median, 'orange', '^')
    ]

    # Create a new figure with specified size
    plt.figure(figsize=(8, 5))

    # Loop through each dataset to plot individually
    for i, (label, ci, median, color, marker) in enumerate(datasets):

        # Calculate asymmetric error bar distances
        lower_err = median - ci[0]  # Distance from median to lower bound
        upper_err = ci[1] - median  # Distance from median to upper bound

        # Plot median point with asymmetric error bars
        plt.errorbar(
            i, median,
            yerr=[[lower_err], [upper_err]],  # asymmetric vertical error bars
            fmt=marker,                       # marker style (circle, square, triangle)
            color=color,                       # marker color
            ecolor=color,                      # error bar color
            capsize=5,                         # length of error bar caps
            markersize=8,                      # size of marker
            elinewidth=2,                      # thickness of error bars
            label=label                        # label for legend
        )

        # Annotate the median value just above the marker
        # f"{median:.2f}" formats the median to 2 decimal places
        plt.text(i, median + 0.01, f"{median:.2f}", ha='center', va='bottom')

    # Set x-axis ticks to dataset names
    plt.xticks(range(3), [d[0] for d in datasets])

    # Label y-axis and add plot title
    plt.ylabel("Median F1")
    plt.title("Median F1 Scores with 95% CI")

    # Add a legend showing which marker/color corresponds to which dataset
    plt.legend()

    # Adjust layout to prevent clipping of labels or legend
    plt.tight_layout()

    # Save figure to disk
    plt.savefig(save_path, bbox_inches='tight')

    # Close the figure to free memory
    plt.close()

    # Print confirmation that plot was saved
    print(f"Error bar plot saved to: {save_path}")


def compare_f1_scores_wilcoxon(label1, label2, f1_scores_1, f1_scores_2):
    """
    Perform a Wilcoxon signed-rank test between two sets of F1 scores
    and calculate the rank-biserial correlation (effect size).

    Parameters
    ----------
    label1 : str
        Name/label for the first dataset.
    label2 : str
        Name/label for the second dataset.
    f1_scores_1 : array-like
        F1 scores for the first dataset.
    f1_scores_2 : array-like
        F1 scores for the second dataset.

    Returns
    -------
    stat : float
        Wilcoxon test statistic.
    p_value : float
        p-value from the Wilcoxon test.
    rbs : float
        Rank-biserial correlation (effect size) between the two datasets.
    """

    # -------------------------
    # Step 1: Wilcoxon signed-rank test
    # -------------------------
    # Tests whether the two paired samples come from the same distribution
    stat, p_value = wilcoxon(f1_scores_1, f1_scores_2)

    # Print test results for clarity
    print(f"{label1} vs. {label2}:")
    if p_value < 0.05:
        print(f"  Statistically significant difference (p = {p_value:.4f} < 0.05).")
    else:
        print(f"  No statistically significant difference (p = {p_value:.4f} >= 0.05).")

    # -------------------------
    # Step 2: Compute rank-biserial correlation (effect size)
    # -------------------------
    # Convert inputs to numpy arrays for element-wise operations
    diffs = np.array(f1_scores_1) - np.array(f1_scores_2)

    # Count number of positive, negative, and zero differences
    n_pos = np.sum(diffs > 0)   # cases where f1_scores_1 > f1_scores_2
    n_neg = np.sum(diffs < 0)   # cases where f1_scores_1 < f1_scores_2
    n_zero = np.sum(diffs == 0) # cases where f1_scores_1 == f1_scores_2

    # Calculate rank-biserial correlation: effect size of difference
    rbs = (n_pos - n_neg) / (n_pos + n_neg + n_zero)

    # Print effect size for clarity
    print(f"  Rank-biserial correlation (effect size): {rbs:.3f}")

    # Add a line for readability
    print()

def main():
    """
    Main pipeline for analyzing F1 scores across Regular, Voronoi, and Synthetic datasets.

    Steps performed:
    1. Parse input F1 scores from JSON strings via command-line arguments.
    2. Compute mean and median F1 scores for each dataset.
    3. Perform bootstrap resampling to estimate medians and 95% confidence intervals.
    4. Plot bootstrap histograms for all datasets in one figure.
    5. Plot median F1 scores with 95% confidence intervals using error bars.
    6. Conduct pairwise Wilcoxon signed-rank tests and compute rank-biserial correlations.
    """

    # -------------------------
    # Step 1: Parse command-line arguments
    # -------------------------
    parser = argparse.ArgumentParser(description="Analyze F1 scores for different datasets")
    parser.add_argument('--regular', required=True, help="JSON string of Regular dataset F1 scores")
    parser.add_argument('--voronoi', required=True, help="JSON string of Voronoi dataset F1 scores")
    parser.add_argument('--synthetic', required=True, help="JSON string of Synthetic dataset F1 scores")
    args = parser.parse_args()

    # -------------------------
    # Step 2: Load F1 scores from JSON strings and convert to float lists
    # -------------------------
    regular_f1_scores = list(map(float, json.loads(args.regular)))
    voronoi_f1_scores = list(map(float, json.loads(args.voronoi)))
    synthetic_f1_scores = list(map(float, json.loads(args.synthetic)))

    # -------------------------
    # Step 3: Compute and print mean F1 scores
    # -------------------------
    find_means("Regular", regular_f1_scores)
    find_means("Voronoi", voronoi_f1_scores)
    find_means("Synthetic", synthetic_f1_scores)
    print()

    # -------------------------
    # Step 4: Compute and print median F1 scores
    # -------------------------
    find_medians("Regular", regular_f1_scores)
    find_medians("Voronoi", voronoi_f1_scores)
    find_medians("Synthetic", synthetic_f1_scores)
    print()

    # -------------------------
    # Step 5: Bootstrap resampling to estimate medians and 95% CIs
    # -------------------------
    regular_boot, regular_boot_median, regular_ci_lower, regular_ci_upper = bootstrap_medians(
        "Regular", regular_f1_scores
    )
    voronoi_boot, voronoi_boot_median, voronoi_ci_lower, voronoi_ci_upper = bootstrap_medians(
        "Voronoi", voronoi_f1_scores
    )
    synthetic_boot, synthetic_boot_median, synthetic_ci_lower, synthetic_ci_upper = bootstrap_medians(
        "Synthetic", synthetic_f1_scores
    )
    print()

    # -------------------------
    # Step 6: Plot bootstrap histograms for all datasets
    # -------------------------
    # Ensure the directory exists
    os.makedirs("results/visuals", exist_ok=True)

    plot_all_bootstraps(
        regular_boot, (regular_ci_lower, regular_ci_upper), regular_boot_median,
        voronoi_boot, (voronoi_ci_lower, voronoi_ci_upper), voronoi_boot_median,
        synthetic_boot, (synthetic_ci_lower, synthetic_ci_upper), synthetic_boot_median,
        save_path="results/visuals/bootstrap_histograms.png"
    )

    # -------------------------
    # Step 7: Plot median F1 scores with 95% confidence intervals (error bars)
    # -------------------------
    plot_all_error_plots(
        (regular_ci_lower, regular_ci_upper), regular_boot_median,
        (voronoi_ci_lower, voronoi_ci_upper), voronoi_boot_median,
        (synthetic_ci_lower, synthetic_ci_upper), synthetic_boot_median,
        save_path="results/visuals/bootstrap_error_plots.png"
    )

    # -------------------------
    # Step 8: Perform pairwise Wilcoxon signed-rank tests and rank-biserial correlations
    # -------------------------
    compare_f1_scores_wilcoxon("Regular", "Voronoi", regular_f1_scores, voronoi_f1_scores)
    compare_f1_scores_wilcoxon("Regular", "Synthetic", regular_f1_scores, synthetic_f1_scores)
    compare_f1_scores_wilcoxon("Voronoi", "Synthetic", voronoi_f1_scores, synthetic_f1_scores)
    print()


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    main()
