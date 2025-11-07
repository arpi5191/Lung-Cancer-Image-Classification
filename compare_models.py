# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--option', nargs='+', required=True)  # Already parses to list
# args = parser.parse_args()
#
# option_list = args.option  # Already a list, no need to split
#
# print("Parsed options:", option_list)

# Create two empty lists to store test results for original and Voronoi types.
# Define a function that takes the patch type as an argument.
# In the function, run a while loop that runs your test, collects test accuracy if the status is "passed," appends it to the list, increments a counter, and eventually returns the list.
# Run the function twice: once for "regular" and once for "voronoi."
# Pass both result lists to a Python script that performs the Wilcoxon test and returns the result.
# Print the Wilcoxon test result on screen.

















# import subprocess
# import os
#
# def main():
#     # Check if the bash script exists
#     script_path = "./run_tumor.sh"
#
#     if not os.path.exists(script_path):
#         print(f"Error: {script_path} not found in current directory")
#         print(f"Current directory: {os.getcwd()}")
#         return
#
#     # Make sure the script is executable
#     if not os.access(script_path, os.X_OK):
#         print(f"Making {script_path} executable...")
#         os.chmod(script_path, 0o755)
#
#     try:
#         print("Running bash script: run_tumor.sh")
#         print("-" * 40)
#
#         result = subprocess.run(['./script.sh'], text=True, check=True)
#
#         # result = subprocess.run(
#         #     ["bash", "run_tumor.sh"],
#         #     text=True,  # This ensures output is returned as string, not bytes
#         #     check=True,  # Raises exception if script fails
#         # )
#
#         print("✅ Script completed successfully!")
#         print(f"Return code: {result.returncode}")
#
#         if result.stdout:
#             print("\n📄 Script Output:")
#             print(result.stdout)
#         else:
#             print("\n📄 No output produced")
#
#         if result.stderr:
#             print("\n⚠️  Script Warnings/Errors:")
#             print(result.stderr)
#
#     except subprocess.CalledProcessError as e:
#         print(f"❌ Script failed with return code: {e.returncode}")
#         if e.stdout:
#             print("\n📄 Script Output:")
#             print(e.stdout)
#         if e.stderr:
#             print("\n❌ Script Errors:")
#             print(e.stderr)
#
#     except subprocess.TimeoutExpired:
#         print("❌ Script timed out after 5 minutes")
#
#     except FileNotFoundError:
#         print("❌ Bash not found. Make sure bash is installed and in PATH")
#
#     except Exception as e:
#         print(f"❌ Unexpected error: {e}")
#
# if __name__ == "__main__":  # Fixed the syntax here
#     main()
#
#


# import subprocess
#
# def run_resnet_model(input_type):
#     try:
#         # Run the ResNetModel.py with the input type
#         result = subprocess.run(
#             ["python", "ResNetModel.py", "--type", input_type],
#             capture_output=True,
#             text=True,
#             check=True
#         )
#
#         # Print entire output (optional, for debugging)
#         print("----- Script Output -----")
#         print(result.stdout)
#         print("-------------------------")
#
#         # Parse accuracy lines from output
#         output_lines = result.stdout.strip().splitlines()
#
#         train_line = next(line for line in output_lines if "Train Accuracies" in line)
#         val_line = next(line for line in output_lines if "Validation Accuracies" in line)
#         test_line = next(line for line in output_lines if "Test Accuracies" in line)
#
#         train_accs = eval(train_line.split(":", 1)[1].strip())
#         val_accs = eval(val_line.split(":", 1)[1].strip())
#         test_acc = eval(test_line.split(":", 1)[1].strip())
#
#         return train_accs[-1], val_accs[-1], test_acc
#
#     except subprocess.CalledProcessError as e:
#         print("Subprocess failed:")
#         print(e.stderr)
#     except Exception as e:
#         print(f"Error: {e}")
#     return None, None, None
#
#
# def main():
#     for input_type in ["tumor", "voronoi"]:
#         print(f"\nRunning for input type: {input_type}")
#         train, val, test = run_resnet_model(input_type)
#         print(f"Final Train Accuracy: {train}")
#         print(f"Final Validation Accuracy: {val}")
#         print(f"Final Test Accuracy: {test}")
#
#
# if __name__ == "__main__":
#     main()





# import subprocess
#
# def run_simulation(script_path, input_type):
#     try:
#         # Run the external script with --type argument
#         result = subprocess.run(
#             ["python", script_path, "--type", input_type],
#             capture_output=True,
#             text=True,
#             check=True
#         )
#
#         # Parse output from stdout
#         output_lines = result.stdout.strip().splitlines()
#         print("\n".join(output_lines))  # Optional: show full output
#
#         # Look for accuracy lines in output
#         train_line = [line for line in output_lines if "Train Accuracies" in line]
#         val_line = [line for line in output_lines if "Validation Accuracies" in line]
#         test_line = [line for line in output_lines if "Test Accuracies" in line]
#
#         train_accs = eval(train_line[0].split(":", 1)[1].strip())
#         val_accs = eval(val_line[0].split(":", 1)[1].strip())
#         test_accs = eval(test_line[0].split(":", 1)[1].strip())
#
#         return train_accs[-1], val_accs[-1], test_accs
#
#     except Exception as e:
#         print(f"Error running simulation with type '{input_type}': {e}")
#         return None, None, None
#
#
# def main():
#     script_path = "ResNetModel.py"  # Adjust if needed
#     input_types = ["tumor", "voronoi"]  # or add "nuclei" etc.
#
#     for input_type in input_types:
#         print(f"\nRunning simulation for: {input_type}")
#         train, val, test = run_simulation(script_path, input_type)
#         print(f"Final Train Accuracy: {train}")
#         print(f"Final Validation Accuracy: {val}")
#         print(f"Final Test Accuracy: {test}")
#
#
# if __name__ == "__main__":
#     main()







# from ResNetModel import run_model
#
# def run_simulations(simulations_num, input_type):
#     simulations_count = 0
#     final_train_accs = []
#     final_val_accs = []
#     final_test_accs = []
#
#     while simulations_count < simulations_num:
#         try:
#             train_accs, val_accs, test_accs = run_model(input_type)
#         except Exception as e:
#             print(f"Exception in simulation {simulations_count + 1}: {e}")
#             continue
#
#         final_train_accs.append(train_accs[-1])
#         final_val_accs.append(val_accs[-1])
#         final_test_accs.append(test_accs)
#
#         simulations_count += 1
#
#     return final_train_accs, final_val_accs, final_test_accs
#
#
# def main():
#     simulations_num = 1
#
#     # For tumor model
#     print("Original Model:")
#     original_train_accs, original_val_accs, original_test_accs = run_simulations(
#         simulations_num, input_type="tumor"
#     )
#     print("Train Accuracies:", original_train_accs)
#     print("Validation Accuracies:", original_val_accs)
#     print("Test Accuracies:", original_test_accs)
#
#     # For Voronoi model
#     print("Voronoi Model:")
#     voronoi_train_accs, voronoi_val_accs, voronoi_test_accs = run_simulations(
#         simulations_num, input_type="voronoi"
#     )
#     print("Train Accuracies:", voronoi_train_accs)
#     print("Validation Accuracies:", voronoi_val_accs)
#     print("Test Accuracies:", voronoi_test_accs)
#
# if __name__ == "__main__":
#     main()
#




# import subprocess
# import os
#
# def run_simulations(simulations_num, bash_script_path):
#     simulations_count = 0
#     final_train_accs = []
#     final_val_accs = []
#     final_test_accs = []
#
#     while True:
#         try:
#             # retcode = os.system("./run_tumor.sh")
#             # print(result.decode('utf-8'))
#             result = subprocess.run(["bash", "./run_tumor.sh"], capture_output=True, text=True, check=True)
#             output = result.stdout.strip()
#             print(output)
#             #
#             # # Example: Parse output here. Adjust if needed.
#             # output = result.stdout.strip().splitlines()
#             # # Assume the last 3 lines are accuracy values
#             # train_accs = eval(output[-3])
#             # val_accs = eval(output[-2])
#             # test_accs = eval(output[-1])
#
#         except Exception as e:
#             print(f"Exception in simulation {simulations_count + 1}: {e}")
#             continue
#
#         # final_train_accs.append(train_accs[-1])
#         # final_val_accs.append(val_accs[-1])
#         # final_test_accs.append(test_accs)
#
#         if simulations_count == simulations_num:
#             break
#
#         simulations_count += 1
#
#     return final_train_accs, final_val_accs, final_test_accs
#
#
# def main():
#     simulations_num = 1
#     bash_script_path = "/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/run_tumor.sh"  # or "run_tumor.sh" if it's in same directory
#
#     original_train_accs, original_val_accs, original_test_accs = run_simulations(
#         simulations_num, bash_script_path
#     )
#
#     print("Original Model:")
#     print("Train Accuracies:", original_train_accs)
#     print("Validation Accuracies:", original_val_accs)
#     print("Test Accuracies:", original_test_accs)
#
# if __name__ == "__main__":
#     main()
#









# # Import packages
# import copy
# import sys
# import subprocess
# import numpy as np
# from scipy import stats
# from scipy.stats import shapiro
# from scipy.stats import wilcoxon
# import ResNetModel as resnet_model
#
# def run_simulations(simulations_num, model, input_type):
#
#     simulations_count = 0
#     final_train_accs = []
#     final_val_accs = []
#     final_test_accs = []
#
#     while simulations_count < simulations_num:
#         try:
#             train_accs, val_accs, test_accs = model(input_type)
#         except Exception as e:
#             print(f"Exception in simulation {simulations_count + 1}: {e}")
#             continue
#
#         final_train_accs.append(train_accs[-1])
#         final_val_accs.append(val_accs[-1])
#         final_test_accs.append(test_accs)
#
#         simulations_count += 1
#
#     return final_train_accs, final_val_accs, final_test_accs
#
#
# # def run_simulations(simulations_num, model, input_type):
# #     """
# #     Run a specified number of simulations using the provided model.
# #
# #     Args:
# #         simulations_num (int): Number of simulations to run.
# #         model (module): The model module containing a 'main' function.
# #         input_type (str): The type of segmentation to use ("tumor", "nuclei", or "voronoi").
# #
# #     Returns:
# #         tuple: Lists containing the final training, validation, and testing accuracies from each simulation.
# #     """
# #
# #     simulations_count = 0
# #     final_train_accs = []
# #     final_val_accs = []
# #     final_test_accs = []
# #
# #     while simulations_count < simulations_num:
# #
# #         try:
# #             if input_type == "tumor":
# #                 train_accs, val_accs, test_accs = subprocess.run(["python", "ResNetModel.py", "--type", "tumor"])
# #             elif input_type == "voronoi":
# #                 train_accs, val_accs, test_accs = subprocess.run(["python", "ResNetModel.py", "--type", "voronoi"])
# #
# #         except Exception as e:
# #             print(f"Exception occurred in simulation {simulations_count + 1}: {e}")
# #             continue
# #
# #         final_train_accs.append(train_accs[-1])
# #         final_val_accs.append(val_accs[-1])
# #         final_test_accs.append(test_accs)
# #
# #         simulations_count += 1
# #
# #     return final_train_accs, final_val_accs, final_test_accs
#
# # def run_simulations(simulations_num, model, input_type):
# #     """
# #     Run a specified number of simulations using the provided model.
# #
# #     Args:
# #         simulations_num (int): Number of simulations to run.
# #         model (module): The model module containing a 'main' function.
# #
# #     Returns:
# #         tuple: Lists containing the final training, validation, and testing accuracies from each simulation.
# #     """
# #
# #     # Initialize the simulation counter to track the number of successful simulations
# #     simulations_count = 0
# #
# #     # Initialize lists to store the final accuracy values from each simulation
# #     final_train_accs = []  # Stores the final training accuracy of each simulation
# #     final_val_accs = []    # Stores the final validation accuracy of each simulation
# #     final_test_accs = []   # Stores the final testing accuracy of each simulation
# #
# #     while True:
# #         try:
# #             # Execute the model's main function and retrieve accuracy lists
# #             # This assumes the 'main' function returns training, validation, and testing accuracy lists
# #             # old_argv = copy.deepcopy(sys.argv)
# #             # sys.argv = ['main', '--type', input_type]
# #             train_accs, val_accs, test_accs = model.main("tumor")
# #             sys.argv = old_argv
# #
# #         except Exception as e:
# #             # If an exception occurs, print it and continue to the next simulation
# #             print(f"Exception occurred in simulation {simulations_count + 1}: {e}")
# #             continue  # Proceed to the next iteration
# #
# #         # Append the last accuracy value from each list to the final results
# #         final_train_accs.append(train_accs[-1])  # Append final training accuracy
# #         final_val_accs.append(val_accs[-1])      # Append final validation accuracy
# #         final_test_accs.append(test_accs)    # Append final testing accuracy
# #
# #         # Increment the simulation counter
# #         simulations_count += 1
# #
# #         # Check if the desired number of simulations has been completed
# #         if simulations_count == simulations_num:
# #             break  # Exit the loop if the required number of simulations is reached
# #
# #     # Return the final accuracy values from all simulations
# #     return final_train_accs, final_val_accs, final_test_accs  # Return final accuracies from all simulations
#
# def summarize_model(name, train, val, test):
#     """
#     Summarizes and prints the performance of a machine learning model based on accuracy metrics.
#
#     This function prints the mean and standard deviation of the training, validation, and test accuracies
#     for the given model, making it easier to evaluate the model's performance across different datasets.
#
#     Parameters:
#     ----------
#     name : str
#         The name of the model (e.g., "Original", "Voronoi").
#
#     train : list or array-like
#         A collection of training accuracies for the model across multiple simulations or epochs.
#
#     val : list or array-like
#         A collection of validation accuracies for the model across multiple simulations or epochs.
#
#     test : list or array-like
#         A collection of test accuracies for the model across multiple simulations or epochs.
#
#     Returns:
#     -------
#     None
#         This function prints the performance metrics directly and does not return any value.
#
#     Example:
#     --------
#     summarize_model("Original", original_train_accs, original_val_accs, original_test_accs)
#     summarize_model("Voronoi", voronoi_train_accs, voronoi_val_accs, voronoi_test_accs)
#     """
#     # Print the name of the model (e.g., "Original" or "Voronoi")
#     print(f"\n{name} Model Performance:")
#
#     # Calculate and print the mean and standard deviation of training accuracy
#     print(f"Train Acc: Mean = {np.mean(train):.4f}, Std = {np.std(train):.4f}")
#
#     # Calculate and print the mean and standard deviation of validation accuracy
#     print(f"Val Acc:   Mean = {np.mean(val):.4f}, Std = {np.std(val):.4f}")
#
#     # Calculate and print the mean and standard deviation of test accuracy
#     print(f"Test Acc:  Mean = {np.mean(test):.4f}, Std = {np.std(test):.4f}")
#
# # Credit to chatGPT: Statistical Tests
#
# def check_normality_differences(original_accs, voronoi_accs, metric_name="Test"):
#     """
#     Check whether the differences in accuracies between two models follow a normal distribution.
#
#     This function uses the Shapiro-Wilk test to assess whether the distribution of paired differences
#     (original - voronoi) is approximately normal. This helps determine whether a parametric test
#     (like a paired t-test) or a non-parametric test (like the Wilcoxon signed-rank test) is appropriate.
#
#     Parameters
#     ----------
#     original_accs : list or array-like
#         Accuracy values from the original model.
#
#     voronoi_accs : list or array-like
#         Accuracy values from the Voronoi-based model.
#
#     metric_name : str, optional (default="Test")
#         A descriptive label for the accuracy metric being tested (e.g., "Test", "Validation", "Train").
#
#     Returns
#     -------
#     test : str
#         The recommended statistical test based on the normality of the differences.
#         Returns "Paired T-Test" if differences are normally distributed, otherwise "Wilcoxon".
#     """
#     # Compute the difference between corresponding accuracy values
#     differences = np.array(original_accs) - np.array(voronoi_accs)
#
#     # Conduct the Shapiro-Wilk test for normality on the differences
#     stat, p = shapiro(differences)
#
#     # Display the test statistic and p-value
#     print(f"\n{metric_name} Accuracy Shapiro-Wilk Test for Normality of Differences:")
#     print(f"statistic: {stat:.4f}, p-value: {p:.4f}")
#
#     # Interpret the p-value to suggest the appropriate statistical test
#     if p < 0.05:
#         # p < 0.05 indicates the data is not normally distributed
#         test = "Wilcoxon"
#         print("Note: Differences are NOT normally distributed (p < 0.05) — consider using Wilcoxon.")
#     else:
#         # p ≥ 0.05 indicates the data is likely normally distributed
#         test = "Paired T-Test"
#         print("Note: Differences appear normally distributed (p ≥ 0.05) — paired t-test is appropriate.")
#
#     # Return the name of the recommended statistical test for further use
#     return test
#
# def perform_wilcoxon_test(original_accs, voronoi_accs, metric_name="Test"):
#     """
#     Perform the Wilcoxon signed-rank test to compare paired accuracy values
#     from two models when the normality assumption is violated.
#
#     This non-parametric test is an alternative to the paired t-test and is used
#     when the differences between paired samples are not normally distributed.
#
#     Parameters
#     ----------
#     original_accs : list or array-like
#         Accuracy values from the original model.
#
#     voronoi_accs : list or array-like
#         Accuracy values from the Voronoi-based model.
#
#     metric_name : str, optional (default="Test")
#         Descriptive name of the accuracy metric being compared.
#
#     Returns
#     -------
#     None
#         Prints the test statistic, p-value, and interpretation of the result.
#     """
#     # Convert inputs to NumPy arrays for vectorized operations
#     original_accs = np.array(original_accs)
#     voronoi_accs = np.array(voronoi_accs)
#
#     # Perform the Wilcoxon signed-rank test on paired samples
#     stat, p = wilcoxon(original_accs, voronoi_accs)
#
#     # Print the results of the test
#     print(f"\n{metric_name} Accuracy Wilcoxon Signed-Rank Test:")
#     print(f"statistic: {stat:.4f}, p-value: {p:.4f}")
#
#     # Interpret the p-value
#     if p < 0.05:
#         print("Result: Statistically significant difference (p < 0.05).")
#     else:
#         print("Result: No statistically significant difference (p ≥ 0.05).")
#
# def perform_paired_ttest(original_accs, voronoi_accs, metric_name="Test"):
#     """
#     Perform a paired t-test between two sets of accuracies and interpret the result.
#
#     Args:
#         original_accs (list or array-like): Accuracies from the original model.
#         voronoi_accs (list or array-like): Accuracies from the Voronoi model.
#         metric_name (str): Name of the accuracy metric (e.g., "Test", "Validation", "Training").
#
#     Returns:
#         None: Prints the t-statistic, p-value, and interpretation.
#     """
#     # Perform the paired t-test to compare the two sets of accuracies
#     t_statistic, p_value = stats.ttest_rel(original_accs, voronoi_accs)
#
#     # Print the results of the t-test
#     print()
#     print(f"{metric_name} Accuracy Paired t-test:")
#     print(f"t-statistic: {t_statistic:.4f}, p-value: {p_value:.4f}")
#
#     # Interpret the p-value to determine statistical significance
#     if p_value < 0.05:
#         print("Result: Statistically significant difference (p < 0.05).")
#     else:
#         print("Result: No statistically significant difference (p ≥ 0.05).")
#
# # def main():
# #     """
# #     Main function to run simulations for both original and Voronoi-based models.
# #     """
# #     simulations_num = 1 # Define the number of simulations to run
# #
# #     # Run simulations for the original model
# #     original_train_accs, original_val_accs, original_test_accs = run_simulations(simulations_num, model, "tumor")
# #
# #     # # Run simulations for the Voronoi-based model
# #     # voronoi_train_accs, voronoi_val_accs, voronoi_test_accs = run_simulations(simulations_num, voronoi_model)
# #
# #     # # Call the function for the "Original" model, passing in its training, validation, and test accuracies
# #     # summarize_model("Original", original_train_accs, original_val_accs, original_test_accs)
# #     #
# #     # # Call the function for the "Voronoi" model, passing in its training, validation, and test accuracies
# #     # summarize_model("Voronoi", voronoi_train_accs, voronoi_val_accs, voronoi_test_accs)
# #     #
# #     # # Determine the appropriate statistical test based on normality of accuracy differences
# #     # # This function returns either "Wilcoxon" or "Paired T-Test" depending on the Shapiro-Wilk test result
# #     # test = check_normality_differences(original_test_accs, voronoi_test_accs)
# #     #
# #     # # Conditionally perform the suitable statistical test
# #     # if test == "Wilcoxon":
# #     #     # If the differences are not normally distributed, use the Wilcoxon signed-rank test (non-parametric)
# #     #     perform_wilcoxon_test(original_test_accs, voronoi_test_accs)
# #     # else:
# #     #     # If the differences are normally distributed, use the paired t-test (parametric)
# #     #     perform_paired_ttest(original_test_accs, voronoi_test_accs)
#
# def main():
#     """
#     Main function to run simulations for both original and Voronoi-based models.
#     """
#     simulations_num = 1
#
#     # Run simulations for the original model using "tumor" patches
#     original_train_accs, original_val_accs, original_test_accs = run_simulations(
#         simulations_num, model=resnet_model, input_type="tumor"
#     )
#
#     print("Original Model:")
#     print("Train Accuracies:", original_train_accs)
#     print("Validation Accuracies:", original_val_accs)
#     print("Test Accuracies:", original_test_accs)
#
#     # Run simulations for the Voronoi model
#     voronoi_train_accs, voronoi_val_accs, voronoi_test_accs = run_simulations(
#         simulations_num, model=resnet_model, input_type="voronoi"
#     )
#
#     print("Voronoi Model:")
#     print("Train Accuracies:", voronoi_train_accs)
#     print("Validation Accuracies:", voronoi_val_accs)
#     print("Test Accuracies:", voronoi_test_accs)
#
# if __name__ == "__main__":
#     main()

# https://medium.com/@ebimsv/ml-series-day-42-statistical-tests-for-model-comparison-4f5cf63da74a
