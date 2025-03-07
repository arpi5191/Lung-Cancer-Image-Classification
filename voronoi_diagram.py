# To run:
    # Run File: python voronoi_diagram.py --nmin 1250 --nmax 10000000 --didx 0 --d 2
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
import skimage.measure
from scipy import ndimage
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.morphology import disk
import skimage.exposure as exposure
from skimage.color import label2rgb
from csbdeep.utils import normalize
from shapely.geometry import Polygon
from skimage.transform import resize
from scipy.stats import skew, kurtosis
from stardist.models import StarDist2D
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skimage.filters import median, threshold_otsu, sobel
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

def filter_on_nuclei_size(labelling, nmin, nmax):

    # Flatten the labelling array into a 1D array and count the occurrence of each label (nucleus)
    # This counts how many pixels are associated with each labeled nucleus
    segmented_cell_sizes = np.bincount(labelling.ravel())

    # Identify nuclei that are either too small or too large based on the given size thresholds (nmin, nmax)
    too_small = segmented_cell_sizes < nmin  # Nuclei smaller than the minimum size
    too_large = segmented_cell_sizes > nmax  # Nuclei larger than the maximum size
    too_small_or_large = too_small | too_large  # Combine both conditions to identify out-of-range nuclei

    # Set the label of nuclei that are out of the specified size range to 0 (remove them)
    labelling[too_small_or_large[labelling]] = 0

    # Return the updated labelling with out-of-range nuclei removed
    return labelling

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

def process_labelling(labelling, nmin, nmax, img, minimum_solidity=0.7):

    centroids = []  # List to store centroids of valid regions
    label_contours = {}  # Dictionary to store contours for each label

    # Get unique labels from the labelling array (i.e., all labeled regions)
    unique_labels = np.unique(labelling)

    # Iterate through each unique label
    for label in unique_labels:
        if label != 0:  # Ignore the background label (assumed to be 0)
            # Create a binary mask for the current label
            binary_mask = np.where(labelling == label, 1, 0).astype(np.uint8)

            # Find contours of the current region
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # If no contours are found, remove the label from the labeled image
            if len(contours) != 1:
                labelling[labelling == label] = 0  # Set all pixels of this label to 0
                continue
            else:
                # Store the contours for the current label
                label_contours[label] = contours

            # Iterate over each contour to validate and calculate centroids
            for contour in contours:
                # Calculate the area of the contour
                contour_area = cv2.contourArea(contour)

                # Calculate the solidity of the contour
                solidity = compute_solidity(contour, contour_area)

                # Debugging: Print contour area and solidity values for inspection
                print(f"Label: {label}, Area: {contour_area}, Solidity: {solidity}")

                # Check if the contour meets the area and solidity criteria
                if contour_area >= nmin and contour_area <= nmax and solidity >= minimum_solidity:
                    # Calculate the centroid using moments
                    M = cv2.moments(contour)
                    if M['m00'] != 0:  # Avoid division by zero
                        cx = int(M['m10'] / M['m00'])  # x-coordinate of the centroid
                        cy = int(M['m01'] / M['m00'])  # y-coordinate of the centroid
                    else:
                        # If moments are zero, calculate centroid as the mean of contour points
                        cx, cy = int(contour[:, :, 0].mean()), int(contour[:, :, 1].mean())

                    # Add the valid centroid to the centroids list
                    centroids.append(np.array([cx, cy]))

    # Return the list of valid centroids and contours for each label
    return centroids, label_contours

def voronoi(tif_paths, classification, patch_dir, nmin, nmax, dapi_channel_idx, downsample_interval):

    # Load the pre-trained StarDist model for nuclei detection
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # Process each .tif file in the provided paths
    for tif_path in tif_paths:

        # Extract the base name of the file for labelling and output directory naming
        basename = tif_path.stem  # Use .stem to get the file name without extension

        # Determine the output directory based on the environment (Docker or local machine)
        if os.path.exists('/.dockerenv'):
            output_dir = f'/voronoi/{classification}/{basename}'  # For Docker environment
        else:
            # For local environment
            output_dir = os.path.join(
                "/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi",
                classification,
                basename,
            )

        # Create the output directory if it doesn't already exist
        os.makedirs(output_dir, exist_ok=True)

        # Confirm successful creation of the output directory
        if os.path.exists(output_dir):
            print(f"Directory '{output_dir}' created successfully.")
        else:
            print(f"Failed to create the directory '{output_dir}'.")

        # Load and downsample the DAPI channel image for nuclei detection
        original_img = load_channel(tif_path, dapi_channel_idx)
        original_img = original_img[::downsample_interval, ::downsample_interval]

        # Preprocessing: Contrast Limited Adaptive Histogram Equalization (CLAHE)
        # Credit to ClaudeAI

        # Create a CLAHE (Contrast Limited Adaptive Histogram Equalization) object
        # clipLimit: Controls the contrast limit, higher values allow more contrast enhancement
        # tileGridSize: The size of the grid for adaptive histogram equalization (larger tiles mean smoother results)
        # Credit to Claude AI: Showing me the CLAHE method
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Normalize the image to uint8 format if it's not already in that format
        # This step scales the pixel values between 0 and 255 if the image is not already of type np.uint8
        if original_img.dtype != np.uint8:
            original_img = ((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255).astype(np.uint8)

        # Apply CLAHE to enhance the local contrast of the image
        original_img = clahe.apply(original_img)

        # Preprocessing: Gaussian Blurring

        # Apply Gaussian Blur to reduce noise and smooth the image
        # (7, 7) is the size of the kernel, and 0 indicates the standard deviation in both x and y directions
        original_img = cv2.GaussianBlur(original_img, (3, 3), 0)

        # Preprocessing: Morphological Closing
        # Credit to ClaudeAI

        # Create a kernel for morphological transformations (a 3x3 matrix of ones)
        # Apply morphological closing to fill small holes in the image (helps in noise reduction)
        # This operation is useful in cleaning up small gaps or holes in the objects
        kernel = np.ones((7, 7), np.uint8)
        original_img = cv2.morphologyEx(original_img, cv2.MORPH_CLOSE, kernel)

        # Perform instance segmentation using the pre-trained StarDist model, scale labelling array
        # to original size, and filter nuclei based on size constraints
        labelling, _ = model.predict_instances(normalize(original_img), prob_thresh=0.5, nms_thresh=0.3)
        labelling = zoom(labelling, downsample_interval, order=0)
        labelling = filter_on_nuclei_size(labelling, nmin, nmax)

        # Merge small adjacent regions using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        labelling_processed = cv2.morphologyEx(labelling.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # Convert back to labeled regions while preserving unique labels
        labelling = skimage.measure.label(labelling_processed)

        # Preprocessing: Watershed Segmentation
        # Credit to ClaudeAI

        # Compute the Euclidean distance transform to highlight nuclei separation
        distance = ndimage.distance_transform_edt(labelling > 0)

        # Identify local maxima in the distance map to use as markers for watershed
        local_max_coords = peak_local_max(distance, min_distance=30, labels=labelling > 0)

        # Create a mask where local maxima will be marked
        local_max_mask = np.zeros(distance.shape, dtype=bool)
        local_max_mask[tuple(local_max_coords.T)] = True

        # Label the local maxima to serve as markers for the watershed algorithm
        markers = ndimage.label(local_max_mask)[0]

        # Apply the watershed algorithm to refine the segmentation of touching nuclei
        refined_labels = watershed(-distance, markers, mask=labelling > 0)

        # Update the original labelling with the refined segmentation
        labelling = refined_labels

        # Skip further processing if no valid nuclei are detected
        if sum(sum(labelling)) == 0:
            continue

        # Save the original image to the output directory
        original_img_title = basename + "_original_image.png"
        original_img_title = os.path.join(output_dir, original_img_title)
        cv2.imwrite(original_img_title, original_img)
        print(f"Original image saved at {original_img_title}")

        # Apply brightness adjustment to the image for better visualization
        brightness_increase = 30  # Value for increasing brightness
        original_bright_img = cv2.convertScaleAbs(original_img, alpha=2, beta=brightness_increase)
        brightened_image_path = os.path.join(output_dir, basename + "_brightened_image.png")
        cv2.imwrite(brightened_image_path, original_bright_img)  # Save the brightened image
        print(f"Brightened image saved at {brightened_image_path}")

        # Visualize and display the labeled image using matplotlib
        fig, ax = plt.subplots(figsize=(8, 8))  # Create a figure and axis with specified size for visualization
        labeled_img = label2rgb(labelling, bg_label=0)  # Generate a color-labeled image from the input labelling
        ax.imshow(labeled_img)  # Display the labeled image on the axis
        ax.axis('off')  # Turn off axis display for cleaner visualization

        # Save the labeled image plot to the specified output directory
        labelling_visualization_path = os.path.join(output_dir, basename + "_labelling_diagram.png")
        plt.savefig(labelling_visualization_path, bbox_inches='tight')  # Save the plot as an image
        plt.close(fig)  # Close the figure to free up memory
        print(f"Labeled diagram saved at {labelling_visualization_path}")

        # Process the labelling to extract centroids and contours of detected regions
        centroids, contours = process_labelling(labelling, nmin, nmax, original_img)

        # Create a blank canvas with the same dimensions as the labelling array for visualizing contours
        contour_canvas = np.zeros_like(labelling, dtype=np.uint8)

        # Draw the contours of each labeled region onto the blank canvas
        for label, contour_list in contours.items():
            # Draw the contours for the current label using white color and a specified thickness
            cv2.drawContours(contour_canvas, contour_list, -1, (255, 255, 255), thickness=4)

        # Define the path to save the contour visualization image
        contours_visualization_path = os.path.join(output_dir, basename + "_contours_visualization.png")
        cv2.imwrite(contours_visualization_path, contour_canvas)

        # Convert the list of centroids into a NumPy array for easier processing
        centroids_array = np.array(centroids)

        # Create a blank canvas with the same dimensions as the labelling array for visualizing centroids
        centroids_canvas = np.zeros_like(labelling, dtype=np.uint8)

        # Loop through each centroid to draw it on the canvas
        for centroid in centroids_array:
            # Convert centroid coordinates to integer values
            x, y = map(int, centroid)

            # Check if the centroid is within the bounds of the canvas
            # Ensure that both x and y are within the image dimensions (width, height)
            if 0 <= x < centroids_canvas.shape[1] and 0 <= y < centroids_canvas.shape[0]:
                # Draw a red filled circle at the centroid location (with color set to white in this case)
                # The color (255, 255, 255) is white, and thickness=-1 means filled circle
                cv2.circle(centroids_canvas, (x, y), radius=5, color=(255, 255, 255), thickness=-1)

        # Define the path and save the centroids visualization image
        centroids_visualization_path = os.path.join(output_dir, basename + "_centroids_visualization.png")
        cv2.imwrite(centroids_visualization_path, centroids_canvas)  # Save the canvas as an image file
        print(f"Centroids visualization saved at {centroids_visualization_path}")  # Output the path of the saved image

        # When generating the Voronoi diagram, set the axis limits to match your image dimensions
        fig, ax = plt.subplots(figsize=(8, 8))
        vor = Voronoi(centroids_array)
        voronoi_plot_2d(vor, ax=ax, show_points=True, show_vertices=True, line_colors="orange", line_width=2)
        ax.scatter(centroids_array[:, 0], centroids_array[:, 1], color='red', s=30, marker="o")

        # Set the limits to match your image dimensions
        img_height, img_width = labelling.shape
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Note: y-axis is flipped in images vs. plots
        ax.axis('off')

        # Define the output file path for saving the Voronoi diagram
        voronoi_diagram_path = os.path.join(output_dir, basename + "_voronoi_diagram.png")
        fig.savefig(voronoi_diagram_path, bbox_inches='tight', dpi=500)
        plt.close(fig)

        # Print the confirmation message indicating the saved Voronoi diagram
        print(f"Voronoi diagram saved at {voronoi_diagram_path}")

        # Open the images as RGBA for overlaying
        original_image = Image.open(brightened_image_path).convert("RGBA")
        labelling_visualization = Image.open(labelling_visualization_path).convert("RGBA")
        contours_visualization = Image.open(contours_visualization_path).convert("RGBA")
        centroids_visualization = Image.open(centroids_visualization_path).convert("RGBA")
        voronoi_diagram = Image.open(voronoi_diagram_path).convert("RGBA")

        # Resize the labelling visualization image to match the original image size, if necessary
        if original_image.size != labelling_visualization.size:
            print(f"Resizing labelling visualization image from {labelling_visualization.size} to {original_image.size}")
            labelling_visualization = labelling_visualization.resize(original_image.size)

        # Combine the original bright image with the labeled image and save the result
        original_labelled_img = np.hstack((original_image, labelling_visualization))
        final_output_dir = os.path.join(output_dir, basename + "_original_labelled_image.png")
        cv2.imwrite(final_output_dir, original_labelled_img)

        # Resize the contours visualization image to match the labelling visualization size, if necessary
        if labelling_visualization.size != contours_visualization.size:
            print(f"Resizing contours visualization image from {contours_visualization.size} to {labelling_visualization.size}")
            contours_visualization = contours_visualization.resize(labelling_visualization.size)

        # Combine the labelling visualization and contours visualization images side by side
        labelling_contours_img = np.hstack((labelling_visualization, contours_visualization))
        final_output_dir = os.path.join(output_dir, basename + "_labelled_contours_image.png")
        cv2.imwrite(final_output_dir, labelling_contours_img)

        # Overlay centroids_visualization on original_image
        if original_image.size != centroids_visualization.size:
            print(f"Resizing centroids visualization image from {centroids_visualization.size} to {original_image.size}")
            centroids_visualization = centroids_visualization.resize(original_image.size)

        # Change the non-zero pixels in the centroids visualization to red
        centroids_visualization.putdata([(0, 0, 255, 255) if pixel[0] != 0 else pixel for pixel in centroids_visualization.getdata()])

        # Combine the contours visualization and centroids visualization images side by side
        original_centroids_img = Image.blend(original_image, centroids_visualization, alpha=0.4)
        final_output_dir = os.path.join(output_dir, basename + "_original_centroids_image.png")
        original_centroids_img.save(final_output_dir, format="PNG")

        # Resize the centroids visualization image to match the contours visualization size, if necessary
        if contours_visualization.size != centroids_visualization.size:
            print(f"Resizing centroids visualization image from {centroids_visualization.size} to {contours_visualization.size}")
            centroids_visualization = centroids_visualization.resize(contours_visualization.size)

        # Combine the contours visualization and centroids visualization images side by side
        contours_centroids_img = Image.blend(contours_visualization, centroids_visualization, alpha=0.4)
        final_output_dir = os.path.join(output_dir, basename + "_contours_centroids_image.png")
        contours_centroids_img.save(final_output_dir, format="PNG")

        # Resize the Voronoi diagram image to match the centroids visualization size, if necessary
        if centroids_visualization.size != voronoi_diagram.size:
            print(f"Resizing Voronoi diagram image from {voronoi_diagram.size} to {centroids_visualization.size}")
            voronoi_diagram = voronoi_diagram.resize(centroids_visualization.size)

        # Blend the centroids visualization and Voronoi diagram images with alpha blending
        voronoi_centroids_img = Image.blend(centroids_visualization, voronoi_diagram, alpha=0.2)
        final_output_dir = os.path.join(output_dir, basename + "_voronoi_centroids_image.png")
        voronoi_centroids_img.save(final_output_dir, format="PNG")

        # Resize the Voronoi diagram image to match the original bright image size, if necessary
        if original_image.size != voronoi_diagram.size:
            print(f"Resizing Voronoi diagram image from {voronoi_diagram.size} to {original_image.size}")
            voronoi_diagram = voronoi_diagram.resize(original_image.size)

        # Blend the original bright image and Voronoi diagram with alpha bending
        original_voronoi_img = Image.blend(original_image, voronoi_diagram, alpha=0.3)
        final_output_dir = os.path.join(output_dir, basename + "_original_voronoi_image.png")
        original_voronoi_img.save(final_output_dir, format="PNG")

        # Convert the original Voronoi diagram image to a NumPy array for processing
        original_voronoi_array = np.array(original_voronoi_img)

        # Extract patches from the Voronoi diagram and save them in the specified directory
        extract_patches(original_voronoi_array, patch_dir, classification, basename)

def extract_patches(image, patch_dir, classification, basename, patch_size=256, stride=256, save_size=512):

    # Determine the output directory based on the environment (Docker or local)
    if os.path.exists('/.dockerenv'):
        patch_output_dir = f'/voronoi_seg/{classification}/{basename}'  # Docker environment path
    else:
        patch_output_dir = os.path.join(patch_dir, basename)  # Local environment path

    # Create the output directory if it doesn't exist
    os.makedirs(patch_output_dir, exist_ok=True)

    # Verify that the directory was successfully created
    if os.path.exists(patch_output_dir):
        print(f"Directory '{patch_output_dir}' was created successfully.")
    else:
        print(f"Failed to create the directory '{patch_output_dir}'.")

    # Get the dimensions of the input image
    h, w = image.shape[:2]

    # Initialize patch label counter
    label = 1

    # Iterate over the width of the image, extracting patches with the given stride
    for x in range(0, w + 1 - stride, stride):

        # Extract a patch from the image
        patch = image[0:h, x:x + patch_size]
        print(0, h, x, x + patch_size)  # Debugging statement to show patch coordinates

        # Resize the patch to the specified save size
        patch = cv2.resize(patch, (save_size, save_size), interpolation=cv2.INTER_LINEAR)

        # Define the patch filename
        patch_filename = f'{basename}_label{label}.tif'

        # Construct the full path for saving the patch
        full_patch_path = os.path.join(patch_output_dir, patch_filename)

        # Save the extracted patch
        cv2.imwrite(full_patch_path, patch)

        # Increment label for the next patch
        label += 1

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

    # Determine the output directory for cancerous patches based on the environment (Docker or local)
    if os.path.exists('/.dockerenv'):
        cancer_patch_dir = '/voronoi/Cancerous'
    else:
        cancer_patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi/Cancerous'

    # Create the directory if it does not exist
    os.makedirs(cancer_patch_dir, exist_ok=True)

    # Verify successful creation of the directory
    if os.path.exists(cancer_patch_dir):
        print(f"Directory '{cancer_patch_dir}' was created successfully.")
    else:
        print(f"Failed to create the directory '{cancer_patch_dir}'.")

    # Determine the output directory for non-cancerous patches based on the environment (Docker or local)
    if os.path.exists('/.dockerenv'):
        no_cancer_patch_dir = '/voronoi/NotCancerous'
    else:
        no_cancer_patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi/NotCancerous'

    # Create the directory if it does not exist
    os.makedirs(no_cancer_patch_dir, exist_ok=True)

    # Verify successful creation of the directory
    if os.path.exists(no_cancer_patch_dir):
        print(f"Directory '{no_cancer_patch_dir}' was created successfully.")
    else:
        print(f"Failed to create the directory '{no_cancer_patch_dir}'.")

    # Determine the output directory for segmented cancerous patches
    if os.path.exists('/.dockerenv'):
        cancer_patch_dir = '/voronoi_seg/Cancerous'
    else:
        cancer_patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_seg/Cancerous'

    # Delete the directory if it exists to ensure a fresh start
    if os.path.exists(cancer_patch_dir):
        shutil.rmtree(cancer_patch_dir)

    # Create a new directory
    os.makedirs(cancer_patch_dir, exist_ok=True)

    # Verify successful creation of the directory
    if os.path.exists(cancer_patch_dir):
        print(f"Directory '{cancer_patch_dir}' was created successfully.")
    else:
        print(f"Failed to create the directory '{cancer_patch_dir}'.")

    # Determine the output directory for segmented non-cancerous patches
    if os.path.exists('/.dockerenv'):
        no_cancer_patch_dir = '/voronoi_seg/NotCancerous'
    else:
        no_cancer_patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_seg/NotCancerous'

    # Delete the directory if it exists to ensure a fresh start
    if os.path.exists(no_cancer_patch_dir):
        shutil.rmtree(no_cancer_patch_dir)

    # Create a new directory
    os.makedirs(no_cancer_patch_dir, exist_ok=True)

    # Verify successful creation of the directory
    if os.path.exists(no_cancer_patch_dir):
        print(f"Directory '{no_cancer_patch_dir}' was created successfully.")
    else:
        print(f"Failed to create the directory '{no_cancer_patch_dir}'.")

    # Process cancerous and non-cancerous image data using the voronoi function and update the DataFrame
    voronoi(cancer_tif_paths, "Cancerous", cancer_patch_dir, args.nmin, args.nmax, args.didx, args.d)
    voronoi(no_cancer_tif_paths, "NotCancerous", no_cancer_patch_dir, args.nmin, args.nmax, args.didx, args.d)

if __name__ == "__main__":
    main()

# Links:
   # https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_stardist.html
