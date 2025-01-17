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
from PIL import Image
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from csbdeep.utils import normalize
import skimage.exposure as exposure
from shapely.geometry import Polygon
from stardist.models import StarDist2D
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)

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

def filter_on_nuclei_size(labeling, nmin, nmax):

    # Flatten the labeling array into a 1D array and count the occurrence of each label (nucleus)
    # This counts how many pixels are associated with each labeled nucleus
    segmented_cell_sizes = np.bincount(labeling.ravel())

    # Identify nuclei that are either too small or too large based on the given size thresholds (nmin, nmax)
    too_small = segmented_cell_sizes < nmin  # Nuclei smaller than the minimum size
    too_large = segmented_cell_sizes > nmax  # Nuclei larger than the maximum size
    too_small_or_large = too_small | too_large  # Combine both conditions to identify out-of-range nuclei

    # Set the label of nuclei that are out of the specified size range to 0 (remove them)
    labeling[too_small_or_large[labeling]] = 0

    # Return the updated labeling with out-of-range nuclei removed
    return labeling

def distance(x1, x2, y1, y2):

    # Calculate the squared difference in x-coordinates
    x_diff = (x2 - x1) ** 2

    # Calculate the squared difference in y-coordinates
    y_diff = (y2 - y1) ** 2

    # Compute the Euclidean distance using the Pythagorean theorem
    dist = math.sqrt(x_diff + y_diff)

    # Return distance
    return dist

def compute_solidity(contour, contour_area):

    # Find the convex hull, which is the smallest convex shape that can fully enclose the contour
    hull = cv2.convexHull(contour)

    # Calculate the area of the convex hull
    hull_area = cv2.contourArea(hull)

    # Compute solidity: ratio of the area of the actual contour to the area of the convex hull
    # Solidity = (contour area) / (convex hull area)
    # If the convex hull area is zero (which can happen if the contour is degenerate), set solidity to 0 to avoid division by zero
    solidity = contour_area / hull_area if hull_area > 0 else 0

    # Return the computed solidity value
    return solidity

def process_labeling(labeling, nmin, nmax, img, minimum_solidity = 0.8):

    # Initialize variables to count contours and store centroids
    contour_count = 0
    centroids = []

    # Get unique labels from the labeling array (i.e., nuclei labels)
    unique_labels = np.unique(labeling)

    # Iterate through each unique label
    for label in unique_labels:
        # Ignore the background label (0)
        if label != 0:

            # Create a binary mask for the current label (1 for the current region, 0 elsewhere)
            binary_mask = np.where(labeling == label, 1, 0).astype(np.uint8)

            # Find contours in the binary mask using OpenCV's findContours method
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Ensure at least one contour is found
            if len(contours) == 0:
                continue

            # Calculate the area of the first contour (there should only be one contour per region)
            contour_area = cv2.contourArea(contours[0])

            # Calculate the solidity of the contour (ratio of contour area to convex hull area)
            solidity = compute_solidity(contours[0], contour_area)

            # Check if the contour meets the size and solidity criteria
            if contour_area < nmin or contour_area > nmax or solidity < minimum_solidity:
                # If the contour doesn't meet the criteria, mark the region as background (0)
                labeling[labeling == label] = 0
            else:
                # Credit to ChatGPT: Providing this calculation
                # If valid, compute the moments to find the centroid
                M = cv2.moments(contours[0])

                # Calculate the centroid (cx, cy) using the moments (m10/m00 and m01/m00)
                if M['m00'] != 0:  # Ensure no division by zero
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Add the valid centroid to the centroids list
                    centroids.append((cx, cy))

                # Increment the contour count
                contour_count += 1

    # Return the total count of valid contours and the list of centroids
    return contour_count, centroids

def voronoi(tif_paths, classification, nmin, nmax, dapi_channel_idx, downsample_interval):

    # Load the pre-trained StarDist model for nuclei detection
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # Process each .tif file
    for tif_path in tif_paths:

        # Extract the base file name (without extension) for output directory naming
        basename = tif_path.name.rstrip('.tif')

        # Determine output directory based on the environment (Docker or local)
        if os.path.exists('/.dockerenv'):
            output_dir = f'/voronoi/{classification}/{basename}'  # Docker environment
        else:
            output_dir = f'/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi/{classification}/{basename}'  # Local machine

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Confirm directory creation
        if os.path.exists(output_dir):
            print(f"Directory '{output_dir}' created successfully.")
        else:
            print(f"Failed to create the directory '{output_dir}'.")

        # Load the DAPI channel image for nuclei segmentation
        img = load_channel(tif_path, dapi_channel_idx)

        # Downsample the image by keeping every nth pixel, specified by downsample_interval
        img = img[::downsample_interval, ::downsample_interval]

        # Use the StarDist model to predict nuclei instances
        labeling, _ = model.predict_instances(normalize(img))

        # Downsample the segmentation labels to match the downsampling of the image
        labeling = zoom(labeling, downsample_interval, order=0)

        # Filter nuclei based on size (keep only within the range [nmin, nmax])
        labeling = filter_on_nuclei_size(labeling, nmin, nmax)

        # Skip further processing if no nuclei are detected
        if sum(sum(labeling)) == 0:
            continue

        # Save the downsampled DAPI image
        image_file = os.path.join(output_dir, "voronoi_image.png")
        cv2.imwrite(image_file, img)
        print(f"Image saved at {image_file}")

        # Process nuclei to extract contours and calculate centroids
        contour_count, centroids = process_labeling(labeling, nmin, nmax, img)

        print(f"Processing file: {tif_path}")
        print(f"Number of valid contours (nuclei): {contour_count}")
        print()

        # Generate a Voronoi diagram using the centroids
        # Credit to ChatGPT: Showing me the general idea for how to generate voronoi diagrams
        vor = Voronoi(centroids)

        # Plot the Voronoi diagram for visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, alpha=0.2)
        ax.axis('off')  # Hide axis labels and ticks

        # Save the Voronoi diagram plot
        final_output_dir = output_dir + "/voronoi_diagram.png"
        fig.savefig(final_output_dir, bbox_inches='tight', dpi=300)  # High DPI for better quality
        plt.close(fig)  # Close figure to release memory

        # Load the tumor image (base) and the Voronoi overlay
        tumor_image_path = output_dir + "/voronoi_image.png"
        overlay_image_path = output_dir + "/voronoi_diagram.png"
        tumor_image = Image.open(tumor_image_path).convert("RGBA")
        overlay_image = Image.open(overlay_image_path).convert("RGBA")

        # Ensure the overlay image matches the base image size
        if overlay_image.size != tumor_image.size:
            print(f"Resizing overlay image from {overlay_image.size} to {tumor_image.size}")
            overlay_image = overlay_image.resize(tumor_image.size)

        # Blend the tumor image and the Voronoi overlay with transparency
        # Credit to ChatGPT: Showing me how to overlay the images
        blended_image = Image.blend(tumor_image, overlay_image, alpha=0.4)

        # Save the final blended image
        output_image_path = output_dir + "/blended_image.png"
        blended_image.save(output_image_path, format="PNG")

        # Print message if the image is saved successfully
        print(f"Output image saved successfully at {output_image_path}")

def main():

    # Define and parse the command-line arguments for input parameters
    parser = argparse.ArgumentParser(description="Segmentation parameters for nuclei detection in TIFF images.")
    parser.add_argument('--nmin', '--nuclei-min-pixels', type=int, help="Minimum number of pixels for a valid nucleus.")
    parser.add_argument('--nmax', '--nuclei-max-pixels', type=int, help="Maximum number of pixels for a valid nucleus.")
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

    # Process cancerous and non-cancerous image data using the voronoi function and update the DataFrame
    voronoi(cancer_tif_paths, "Cancerous", args.nmin, args.nmax, args.didx, args.d)
    voronoi(no_cancer_tif_paths, "NotCancerous", args.nmin, args.nmax, args.didx, args.d)

if __name__ == "__main__":
    main()
