# To run:
    # Run File: python voronoi.py --didx 0 --d 2
    # Create image: docker build . -t voronoi
    # Get a shell into a container: docker run -it -v /Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi:/voronoi voronoi bash
    # Run Segmentation: docker run -v /Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi:/voronoi -v /Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/:/ -v /Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi:/voronoi

# Import packages
import os
import cv2
import math
import time
import random
import shutil
import pathlib
import tifffile
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from csbdeep.utils import normalize
import skimage.exposure as exposure
from shapely.geometry import Polygon
from stardist.models import StarDist2D
from scipy.stats import skew, kurtosis
from scipy.spatial import Voronoi, voronoi_plot_2d

def load_channel(tif_path, channel_idx):
    ''' Load the specified channel from a multi-channel TIFF file as a 2D numpy array.

    Args:
        - tif_path: str, the file path to the multi-channel TIFF image.
        - channel_idx: int, the 0-based index of the desired channel (e.g., DAPI is often at index 0).

    Returns:
        - img: 2D numpy array, the selected channel of the image.
    '''

    # Read the desired channel from the TIFF file
    channel = tifffile.imread(tif_path)

    # Rescale the intensity values of the image to a range of 0-255 for consistency
    return exposure.rescale_intensity(channel, in_range='image', out_range=(0, 255)).astype(np.uint8)

def voronoi(tif_paths, classification, dapi_channel_idx, downsample_interval, voronoi_data):

    # Iterate through each .tif file path
    for tif_path in tif_paths:

        # Extract the base name of the file (without extension) for labeling purposes
        basename = tif_path.name.rstrip('.tif')
        # print("Segmenting tif path: {}".format(basename))

        # Check if the script is running in a Docker environment
        if os.path.exists('/.dockerenv'):
            output_dir = f'/voronoi/{classification}/{basename}'  # Path for Docker
        else:
            # Path for local machine
            output_dir = f'/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi/{classification}/{basename}'

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Confirm if the directory was created successfully
        if os.path.exists(output_dir):
            print(f"Directory '{output_dir}' created successfully.")
        else:
            print(f"Failed to create the directory '{output_dir}'.")

        # Load and process the DAPI channel image
        # Downsample the image by taking every nth pixel, as specified by downsample_interval
        img = load_channel(tif_path, dapi_channel_idx)
        img = img[::downsample_interval, ::downsample_interval]

        num_centroids = 10  # Number of centroids for Voronoi diagram
        img_height, img_width = img.shape  # Get the dimensions of the image

        # Randomly generate centroids within the image dimensions
        # Credit to chatGPT: General Idea
        centroids = np.array([[random.randint(0, img_width), random.randint(0, img_height)] for _ in range(num_centroids)])

        # Create a Voronoi object using the centroids
        vor = Voronoi(centroids)

        # Plot the Voronoi diagram for visualization
        # Credit to ChatGPT: Providing code to generate plot
        fig, ax = plt.subplots(figsize=(8, 8))
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange')  # Plot the Voronoi diagram with orange lines
        ax.set_title("Voronoi Diagram")  # Set the title of the plot
        ax.axis('off')  # Turn off axis labels and ticks

        # Save the plot as a PNG file
        final_output_dir = output_dir + "/voronoi_diagram"
        fig.savefig(final_output_dir, bbox_inches='tight', dpi=300)  # Save the figure with high DPI for quality
        plt.close(fig)  # Close the figure after saving to free memory

        # Generate a random RGB color for each centroid
        colors = [np.random.randint(0, 256, 3) for _ in range(len(centroids))]

        # Initialize the Voronoi image as a 3D array (height, width, 3) to store RGB values
        voronoi_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)  # Ensure 3 channels for RGB

        # Iterate through every pixel in the image
        # Credit to ChatGPT: General Idea
        for x in range(img_width):
            for y in range(img_height):
                shortest_dist = float('inf')  # Initialize the shortest distance to infinity
                position = -1  # Initialize the position of the closest centroid

                # Find the closest centroid to the current pixel
                for i, coord in enumerate(centroids):
                    dist = distance(x, coord[0], y, coord[1])  # Calculate the distance to the centroid
                    if dist < shortest_dist:  # Update the shortest distance if a closer centroid is found
                        shortest_dist = dist
                        position = i  # Update the position to the index of the closest centroid

                # Assign the closest centroid's color to the pixel
                voronoi_img[y, x] = colors[position]  # Store the color associated with the closest centroid for each pixel

        # Save the Voronoi image as a PNG file
        final_output_dir = output_dir + "/voronoi_image.png"  # Add .png extension to the file name

        # If the image is in RGB, convert it to BGR before saving with OpenCV
        cv2.imwrite(final_output_dir, cv2.cvtColor(voronoi_img, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR before saving

        # Credit to ChatGPT: General Idea
        # Iterate over each region in the Voronoi diagram
        for region in vor.regions:

            # Skip empty regions or regions that contain '-1', which indicates unbounded regions
            if region == [] or -1 in region:
                continue

            # Retrieve the coordinates of the vertices for the current region
            # The 'region' variable contains indices that map to the vertices in 'vor.vertices'
            coords = [vor.vertices[i] for i in region]

            # Create a polygon using the coordinates of the vertices for the current Voronoi region
            # The Shapely library's Polygon class is used to facilitate feature extraction and analysis
            polygon = Polygon(coords)

            # Calculate the area of the polygon
            area = polygon.area

            # Calculate the perimeter (length) of the polygon
            perimeter = polygon.length

            # Calculate the compactness of the polygon:
            # This measures how circular the polygon is, where a value of 1 means it is perfectly circular.
            # The formula is: compactness = (perimeter^2) / (4 * pi * area)
            compactness = (perimeter * perimeter) / (4 * math.pi * area)

            # Calculate the circularity of the polygon:
            # Circularity measures the "roundness" of the polygon, where a value of 1 indicates a perfect circle.
            # The formula is: circularity = (4 * pi * area) / (perimeter^2)
            circularity = (4 * math.pi * area) / (perimeter * perimeter)

            # Calculate the number of vertices (corner points) in the polygon
            # The number of vertices is determined by the length of the 'coords' list
            num_vertices = len(coords)

            # Separate the x and y coordinates from the vertices
            # 'coords' contains (x, y) pairs, which are unzipped into two separate lists: x_coords and y_coords
            x_coords, y_coords = zip(*coords)

            # Calculate the skewness of the x and y coordinates
            # Skewness measures the asymmetry of the distribution of the coordinates
            # Positive skewness indicates a longer tail on the right, negative skewness indicates a longer tail on the left
            # This could help identify irregularities in the polygon's shape
            x_skewness, y_skewness = skew(x_coords), skew(y_coords)

            # Calculate the kurtosis of the x and y coordinates
            # Kurtosis measures the "tailedness" of the distribution of the coordinates
            # High kurtosis indicates heavy tails, while low kurtosis indicates light tails
            x_kurtosis, y_kurtosis = kurtosis(x_coords), kurtosis(y_coords)

            # Define a list called 'features' that contains the calculated metrics for the current Voronoi polygon.
            # Each element in this list corresponds to one of the features we've computed (e.g., area, perimeter, skewness, etc.)
            # These features will be used to populate a row in the DataFrame for storing and further analysis.
            features = [area, perimeter, compactness, circularity, num_vertices,
                        x_skewness, y_skewness, x_kurtosis, y_kurtosis, classification]

            # Append the calculated 'features' as a new row in the 'voronoi_data' DataFrame
            voronoi_data.loc[len(voronoi_data)] = features

    # Return the DataFrame containing the Voronoi features
    return voronoi_data


def distance(x1, x2, y1, y2):

    # Calculate the squared difference in x-coordinates
    x_diff = (x2 - x1) ** 2

    # Calculate the squared difference in y-coordinates
    y_diff = (y2 - y1) ** 2

    # Compute the Euclidean distance using the Pythagorean theorem
    dist = math.sqrt(x_diff + y_diff)

    # Return distance
    return dist

def main():

    # Define and parse the command-line arguments for input parameters
    parser = argparse.ArgumentParser(description="Parameters for voronoi diagrams and images in TIFF images.")
    parser.add_argument('--didx', '--dapi-channel-idx', type=int, help="Index of the DAPI channel (typically 0).")
    parser.add_argument('--d', '--downsample-interval', type=int, help="Factor by which to downsample the image.")
    args = parser.parse_args()

    # Set paths for Cancerous and NotCancerous TIFF directories based on environment
    if os.path.exists('/.dockerenv'):
        cancer_tif_dir = '/tif/Cancerous'
        no_cancer_tif_dir = '/tif/NotCancerous'
    else:
        cancer_tif_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif/Cancerous'
        no_cancer_tif_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif/NotCancerous'

    # Verify the Cancerous TIFF directory exists
    if not pathlib.Path(cancer_tif_dir).exists():
        raise FileNotFoundError(f"TIFF directory '{cancer_tif_dir}' does not exist. Please check the path.")

    # List all TIFF files in the Cancerous directory
    cancer_tif_paths = list(pathlib.Path(cancer_tif_dir).glob('*.tif'))
    print(f"Cancerous TIFF Paths: {cancer_tif_paths}")
    if not cancer_tif_paths:
        raise FileNotFoundError(f"The directory '{cancer_tif_dir}' does not contain any .tif files.")

    # Verify the NotCancerous TIFF directory exists
    if not pathlib.Path(no_cancer_tif_dir).exists():
        raise FileNotFoundError(f"TIFF directory '{no_cancer_tif_dir}' does not exist. Please check the path.")

    # List all TIFF files in the NotCancerous directory
    no_cancer_tif_paths = list(pathlib.Path(no_cancer_tif_dir).glob('*.tif'))
    print(f"NotCancerous TIFF Paths: {no_cancer_tif_paths}")
    if not no_cancer_tif_paths:
        raise FileNotFoundError(f"The directory '{no_cancer_tif_dir}' does not contain any .tif files.")

    # Check if the script is running in a Docker environment and set the output directory path accordingly
    if os.path.exists('/.dockerenv'):
        voronoi_dir = '/voronoi'  # Output path for Docker environment
    else:
        voronoi_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi'  # Output path for local environment

    # Check if the output directory already exists
    if os.path.exists(voronoi_dir ):
        # Remove the existing directory and all its contents
        shutil.rmtree(voronoi_dir)
        print(f"Directory '{voronoi_dir}' has been deleted.")
    else:
        print(f"Directory '{voronoi_dir}' does not exist.")

    # Create the output directory if it doesn't exist
    os.makedirs(voronoi_dir, exist_ok=True)

    # Confirm the creation of the output directory and print the status
    if os.path.exists(voronoi_dir):
        print(f"Directory '{voronoi_dir}' was created successfully.")
    else:
        print(f"Failed to create the directory '{voronoi_dir}'.")

    # Set the output directory for voronoi diagrams and images of cancerous images based on the environment (Docker or local)
    if os.path.exists('/.dockerenv'):
        cancer_patch_dir = '/voronoi/Cancerous'  # Output path for Docker environment
    else:
        cancer_patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi/Cancerous'  # Output path for local environment

    # Create the cancerous voronoi directory if it doesn't already exist
    os.makedirs(cancer_patch_dir, exist_ok=True)

    # Verify the creation of the cancerous voronoi directory and print its status
    if os.path.exists(cancer_patch_dir):
        print(f"Directory '{cancer_patch_dir}' was created successfully.")
    else:
        print(f"Failed to create the directory '{cancer_patch_dir}'.")

    # Set the output directory for voronoi diagrams and images of non-cancerous images based on the environment (Docker or local)
    if os.path.exists('/.dockerenv'):
        no_cancer_patch_dir = '/voronoi/NotCancerous'  # Output path for Docker environment
    else:
        no_cancer_patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi/NotCancerous'  # Output path for local environment

    # Create the non-cancerous patch directory if it doesn't already exist
    os.makedirs(no_cancer_patch_dir, exist_ok=True)

    # Verify the creation of the non-cancerous patch directory and print its status
    if os.path.exists(no_cancer_patch_dir):
        print(f"Directory '{no_cancer_patch_dir}' was created successfully.")
    else:
        print(f"Failed to create the directory '{no_cancer_patch_dir}'.")

    # Define the column names for the DataFrame
    # These columns correspond to the various features of the Voronoi polygons that we want to calculate and store
    # Define the column names for the DataFrame
    # These columns correspond to the various features of the Voronoi polygons that we want to calculate and store.
    # The columns include various geometric and statistical features (e.g., area, perimeter, skewness, kurtosis) of the polygons,
    # which are essential for further analysis and model training. Additionally, a 'Diagnosis' column is added to store the label
    # (e.g., cancerous or non-cancerous) associated with each polygon.
    columns = [
        "Area",                  # The area of the polygon
        "Perimeter",             # The perimeter (length) of the polygon
        "Compactness",           # Compactness: how circular the polygon is
        "Circularity",           # Circularity: how close the polygon is to a perfect circle
        "Number of Vertices",    # The number of vertices in the polygon
        "X Skewness",            # Skewness of the x-coordinates of the polygon vertices
        "Y Skewness",            # Skewness of the y-coordinates of the polygon vertices
        "X Kurtosis",            # Kurtosis of the x-coordinates of the polygon vertices
        "Y Kurtosis",            # Kurtosis of the y-coordinates of the polygon vertices
        "Diagnosis"              # The label indicating whether the polygon is cancerous or non-cancerous
    ]

    # Create an empty DataFrame using the defined column names
    # This DataFrame will store the computed features for each Voronoi region/polygon
    voronoi_data = pd.DataFrame(columns=columns)

    # Process cancerous and non-cancerous image data using the voronoi function and update the DataFrame
    voronoi_data = voronoi(cancer_tif_paths, "Cancerous", args.didx, args.d, voronoi_data)
    voronoi_data = voronoi(no_cancer_tif_paths, "NotCancerous", args.didx, args.d, voronoi_data)

    print(voronoi_data)

if __name__ == "__main__":
    main()
