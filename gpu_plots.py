import pandas as pd
import matplotlib.pyplot as plt

# Load the GPU metrics CSV file into a pandas DataFrame
csv_path = "gpu_metrics/gpu_metrics.csv"
df = pd.read_csv(csv_path)  # df now contains columns like 'Number of GPUs', 'Average Latencies', etc.

# List of metrics to plot
metrics = ["Average Latencies", "Average Throughputs", "Average Memories"]

# Create a figure with 3 subplots arranged in 1 row and 3 columns
# figsize sets the overall figure size (width=18 inches, height=5 inches)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loop through each metric and corresponding subplot
for i, metric in enumerate(metrics):
    # Plot the Number of GPUs vs. Metric in the i-th subplot
    axes[i].plot(df['Number of GPUs'], df[metric], marker='o')

    # Label the x-axis
    axes[i].set_xlabel("Number of GPUs")

    # Label the y-axis with the metric name
    axes[i].set_ylabel(metric)

    # Set the title of the subplot
    axes[i].set_title(f"Number of GPUs vs. {metric}")

    # Enable grid lines for easier reading
    axes[i].grid(True)

# Adjust layout so subplots don't overlap
plt.tight_layout()

# Save the figure as a PNG file in the gpu_metrics folder
plt.savefig("gpu_metrics/gpu_metrics.png")

# Close the figure to free up memory
plt.close()

# Print confirmation that the plot was saved
print("Saved combined GPU metrics plot to gpu_metrics/gpu_metrics.png")
