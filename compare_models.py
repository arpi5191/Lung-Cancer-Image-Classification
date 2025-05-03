# Import packages
import numpy as np
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import wilcoxon
import ResNetModel as original_model
import ResNetModel_voronoi as voronoi_model

def run_simulations(simulations_num, model):
    """
    Run a specified number of simulations using the provided model.

    Args:
        simulations_num (int): Number of simulations to run.
        model (module): The model module containing a 'main' function.

    Returns:
        tuple: Lists containing the final training, validation, and testing accuracies from each simulation.
    """

    # Initialize the simulation counter to track the number of successful simulations
    simulations_count = 0

    # Initialize lists to store the final accuracy values from each simulation
    final_train_accs = []  # Stores the final training accuracy of each simulation
    final_val_accs = []    # Stores the final validation accuracy of each simulation
    final_test_accs = []   # Stores the final testing accuracy of each simulation

    while True:
        try:
            # Execute the model's main function and retrieve accuracy lists
            # This assumes the 'main' function returns training, validation, and testing accuracy lists
            train_accs, val_accs, test_accs = model.main()

        except Exception as e:
            # If an exception occurs, print it and continue to the next simulation
            print(f"Exception occurred in simulation {simulations_count + 1}: {e}")
            continue  # Proceed to the next iteration

        # Append the last accuracy value from each list to the final results
        final_train_accs.append(train_accs[-1])  # Append final training accuracy
        final_val_accs.append(val_accs[-1])      # Append final validation accuracy
        final_test_accs.append(test_accs)    # Append final testing accuracy

        # Increment the simulation counter
        simulations_count += 1

        # Check if the desired number of simulations has been completed
        if simulations_count == simulations_num:
            break  # Exit the loop if the required number of simulations is reached

    # Return the final accuracy values from all simulations
    return final_train_accs, final_val_accs, final_test_accs  # Return final accuracies from all simulations

def summarize_model(name, train, val, test):
    """
    Summarizes and prints the performance of a machine learning model based on accuracy metrics.

    This function prints the mean and standard deviation of the training, validation, and test accuracies
    for the given model, making it easier to evaluate the model's performance across different datasets.

    Parameters:
    ----------
    name : str
        The name of the model (e.g., "Original", "Voronoi").

    train : list or array-like
        A collection of training accuracies for the model across multiple simulations or epochs.

    val : list or array-like
        A collection of validation accuracies for the model across multiple simulations or epochs.

    test : list or array-like
        A collection of test accuracies for the model across multiple simulations or epochs.

    Returns:
    -------
    None
        This function prints the performance metrics directly and does not return any value.

    Example:
    --------
    summarize_model("Original", original_train_accs, original_val_accs, original_test_accs)
    summarize_model("Voronoi", voronoi_train_accs, voronoi_val_accs, voronoi_test_accs)
    """
    # Print the name of the model (e.g., "Original" or "Voronoi")
    print(f"\n{name} Model Performance:")

    # Calculate and print the mean and standard deviation of training accuracy
    print(f"Train Acc: Mean = {np.mean(train):.4f}, Std = {np.std(train):.4f}")

    # Calculate and print the mean and standard deviation of validation accuracy
    print(f"Val Acc:   Mean = {np.mean(val):.4f}, Std = {np.std(val):.4f}")

    # Calculate and print the mean and standard deviation of test accuracy
    print(f"Test Acc:  Mean = {np.mean(test):.4f}, Std = {np.std(test):.4f}")

# Credit to chatGPT: Statistical Tests

def check_normality_differences(original_accs, voronoi_accs, metric_name="Test"):
    """
    Check whether the differences in accuracies between two models follow a normal distribution.

    This function uses the Shapiro-Wilk test to assess whether the distribution of paired differences
    (original - voronoi) is approximately normal. This helps determine whether a parametric test
    (like a paired t-test) or a non-parametric test (like the Wilcoxon signed-rank test) is appropriate.

    Parameters
    ----------
    original_accs : list or array-like
        Accuracy values from the original model.

    voronoi_accs : list or array-like
        Accuracy values from the Voronoi-based model.

    metric_name : str, optional (default="Test")
        A descriptive label for the accuracy metric being tested (e.g., "Test", "Validation", "Train").

    Returns
    -------
    test : str
        The recommended statistical test based on the normality of the differences.
        Returns "Paired T-Test" if differences are normally distributed, otherwise "Wilcoxon".
    """
    # Compute the difference between corresponding accuracy values
    differences = np.array(original_accs) - np.array(voronoi_accs)

    # Conduct the Shapiro-Wilk test for normality on the differences
    stat, p = shapiro(differences)

    # Display the test statistic and p-value
    print(f"\n{metric_name} Accuracy Shapiro-Wilk Test for Normality of Differences:")
    print(f"statistic: {stat:.4f}, p-value: {p:.4f}")

    # Interpret the p-value to suggest the appropriate statistical test
    if p < 0.05:
        # p < 0.05 indicates the data is not normally distributed
        test = "Wilcoxon"
        print("Note: Differences are NOT normally distributed (p < 0.05) — consider using Wilcoxon.")
    else:
        # p ≥ 0.05 indicates the data is likely normally distributed
        test = "Paired T-Test"
        print("Note: Differences appear normally distributed (p ≥ 0.05) — paired t-test is appropriate.")

    # Return the name of the recommended statistical test for further use
    return test

def perform_wilcoxon_test(original_accs, voronoi_accs, metric_name="Test"):
    """
    Perform the Wilcoxon signed-rank test to compare paired accuracy values
    from two models when the normality assumption is violated.

    This non-parametric test is an alternative to the paired t-test and is used
    when the differences between paired samples are not normally distributed.

    Parameters
    ----------
    original_accs : list or array-like
        Accuracy values from the original model.

    voronoi_accs : list or array-like
        Accuracy values from the Voronoi-based model.

    metric_name : str, optional (default="Test")
        Descriptive name of the accuracy metric being compared.

    Returns
    -------
    None
        Prints the test statistic, p-value, and interpretation of the result.
    """
    # Convert inputs to NumPy arrays for vectorized operations
    original_accs = np.array(original_accs)
    voronoi_accs = np.array(voronoi_accs)

    # Perform the Wilcoxon signed-rank test on paired samples
    stat, p = wilcoxon(original_accs, voronoi_accs)

    # Print the results of the test
    print(f"\n{metric_name} Accuracy Wilcoxon Signed-Rank Test:")
    print(f"statistic: {stat:.4f}, p-value: {p:.4f}")

    # Interpret the p-value
    if p < 0.05:
        print("Result: Statistically significant difference (p < 0.05).")
    else:
        print("Result: No statistically significant difference (p ≥ 0.05).")

def perform_paired_ttest(original_accs, voronoi_accs, metric_name="Test"):
    """
    Perform a paired t-test between two sets of accuracies and interpret the result.

    Args:
        original_accs (list or array-like): Accuracies from the original model.
        voronoi_accs (list or array-like): Accuracies from the Voronoi model.
        metric_name (str): Name of the accuracy metric (e.g., "Test", "Validation", "Training").

    Returns:
        None: Prints the t-statistic, p-value, and interpretation.
    """
    # Perform the paired t-test to compare the two sets of accuracies
    t_statistic, p_value = stats.ttest_rel(original_accs, voronoi_accs)

    # Print the results of the t-test
    print()
    print(f"{metric_name} Accuracy Paired t-test:")
    print(f"t-statistic: {t_statistic:.4f}, p-value: {p_value:.4f}")

    # Interpret the p-value to determine statistical significance
    if p_value < 0.05:
        print("Result: Statistically significant difference (p < 0.05).")
    else:
        print("Result: No statistically significant difference (p ≥ 0.05).")

def main():
    """
    Main function to run simulations for both original and Voronoi-based models.
    """
    simulations_num = 20 # Define the number of simulations to run

    # Run simulations for the original model
    original_train_accs, original_val_accs, original_test_accs = run_simulations(simulations_num, original_model)

    # Run simulations for the Voronoi-based model
    voronoi_train_accs, voronoi_val_accs, voronoi_test_accs = run_simulations(simulations_num, voronoi_model)

    # Call the function for the "Original" model, passing in its training, validation, and test accuracies
    summarize_model("Original", original_train_accs, original_val_accs, original_test_accs)

    # Call the function for the "Voronoi" model, passing in its training, validation, and test accuracies
    summarize_model("Voronoi", voronoi_train_accs, voronoi_val_accs, voronoi_test_accs)

    # Determine the appropriate statistical test based on normality of accuracy differences
    # This function returns either "Wilcoxon" or "Paired T-Test" depending on the Shapiro-Wilk test result
    test = check_normality_differences(original_test_accs, voronoi_test_accs)

    # Conditionally perform the suitable statistical test
    if test == "Wilcoxon":
        # If the differences are not normally distributed, use the Wilcoxon signed-rank test (non-parametric)
        perform_wilcoxon_test(original_test_accs, voronoi_test_accs)
    else:
        # If the differences are normally distributed, use the paired t-test (parametric)
        perform_paired_ttest(original_test_accs, voronoi_test_accs)

# Ensure that the main function is called when the script is executed directly
if __name__ == "__main__":
    main()

# https://medium.com/@ebimsv/ml-series-day-42-statistical-tests-for-model-comparison-4f5cf63da74a
