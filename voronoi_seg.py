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
    '''
    Load the specified channel from a multi-channel TIFF file as a 2D numpy array.

    Args:
        tif_path (str): Path to the multi-channel TIFF image.
        channel_idx (int): 0-based index of the desired channel (e.g., DAPI often index 0).

    Returns:
        np.ndarray: 2D array of the selected channel image, intensity scaled to 0-255 as uint8.
    '''

    # Read the entire multi-channel TIFF image (all channels)
    channel = tifffile.imread(tif_path)

    # Rescale the pixel intensity values of the selected channel to range 0-255 and convert to uint8
    return exposure.rescale_intensity(channel, in_range='image', out_range=(0, 255)).astype(np.uint8)

def filter_on_nuclei_size(labelling, nmin, nmax):
    '''
    Filter nuclei labels by size, removing those too small or too large.

    Args:
        labelling (np.ndarray): 2D array with integer labels for each nucleus.
        nmin (int): Minimum allowed nucleus size (pixels).
        nmax (int): Maximum allowed nucleus size (pixels).

    Returns:
        np.ndarray: Updated labeling array with out-of-range nuclei set to 0.
    '''

    # Count pixels per label to measure nucleus size
    segmented_cell_sizes = np.bincount(labelling.ravel())

    # Identify labels with size out of range
    too_small = segmented_cell_sizes < nmin
    too_large = segmented_cell_sizes > nmax
    too_small_or_large = too_small | too_large

    # Remove out-of-range nuclei by setting their labels to 0
    labelling[too_small_or_large[labelling]] = 0

    return labelling

def distance(x1, x2, y1, y2):
    '''
    Compute Euclidean distance between two points (x1, y1) and (x2, y2).

    Args:
        x1, x2, y1, y2 (float): Coordinates of the two points.

    Returns:
        float: Euclidean distance.
    '''

    x_diff = (x2 - x1) ** 2
    y_diff = (y2 - y1) ** 2
    dist = math.sqrt(x_diff + y_diff)
    return dist

def compute_solidity(contour, contour_area):
    '''
    Calculate solidity of a contour: ratio of contour area to convex hull area.

    Args:
        contour (np.ndarray): Contour points of the shape.
        contour_area (float): Precomputed area of the contour.

    Returns:
        float: Solidity (compactness) value between 0 and 1.
    '''

    # Calculate convex hull and its area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    # Avoid division by zero for degenerate contours
    solidity = contour_area / hull_area if hull_area > 0 else 0
    return solidity

def process_labelling(labelling, nmin, nmax, img, minimum_solidity=0.4):
    """
    Process labeled regions in an image to filter valid nuclei based on area and solidity,
    and compute centroids of those valid nuclei.

    Args:
        labelling (np.ndarray): 2D array with integer labels representing segmented regions.
        nmin (int): Minimum allowed area for a nucleus.
        nmax (int): Maximum allowed area for a nucleus.
        img (np.ndarray): Original image (not directly used here but often passed for context).
        minimum_solidity (float): Threshold for minimum solidity to accept a nucleus (0 to 1).

    Returns:
        centroids (list of np.ndarray): List of (x, y) coordinates of valid nuclei centroids.
        label_contours (dict): Dictionary mapping label -> contours of that region.
    """

    centroids = []  # List to store centroids of valid regions
    label_contours = {}  # Dictionary to store contours for each label

    # Extract all unique labels present in the labelling (including background 0)
    unique_labels = np.unique(labelling)

    for label in unique_labels:
        if label != 0:  # Skip background
            # Create a binary mask isolating the current label
            binary_mask = np.where(labelling == label, 1, 0).astype(np.uint8)

            # Find contours for this mask (retrieve full hierarchy)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # If there is not exactly one contour, discard this label as invalid
            if len(contours) != 1:
                labelling[labelling == label] = 0  # Remove this label
                continue
            else:
                label_contours[label] = contours  # Save contour for valid label

            # For each contour (only one expected), validate area and solidity
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                solidity = compute_solidity(contour, contour_area)

                # Debug print for tracking values
                print(f"Label: {label}, Area: {contour_area}, Solidity: {solidity}")

                # Accept contour only if it meets size and solidity criteria
                if nmin <= contour_area <= nmax and solidity >= minimum_solidity:
                    # Compute centroid from image moments, fallback to mean if moments zero
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx, cy = int(contour[:, :, 0].mean()), int(contour[:, :, 1].mean())

                    # Append the centroid coordinates to the list
                    centroids.append(np.array([cx, cy]))

    return centroids, label_contours

def voronoi(tif_paths, classification, intermediates_patch_dir, patch_dir, nmin, nmax, dapi_channel_idx, downsample_interval):
    """
    Processes a list of TIFF images to perform nuclei segmentation, generate Voronoi diagrams,
    and save various visualization outputs including labeled images, contours, centroids, and Voronoi overlays.

    Parameters:
    - tif_paths (list of Path): List of file paths to multi-channel TIFF images.
    - classification (str): Classification label used to organize output directories.
    - patch_dir (str): Directory path to save extracted image patches.
    - nmin (int): Minimum nucleus size (in pixels) to keep.
    - nmax (int): Maximum nucleus size (in pixels) to keep.
    - dapi_channel_idx (int): Index of the DAPI channel in the TIFF images.
    - downsample_interval (int): Factor by which to downsample the input images for processing.

    Process:
    - Load and preprocess DAPI channel images (downsampling, CLAHE, Gaussian blur, morphological closing).
    - Perform nuclei segmentation with StarDist and refine with watershed.
    - Filter segmented nuclei based on size constraints.
    - Extract and save visualization images: original, brightened, labeled, contours, centroids.
    - Generate and save Voronoi diagrams based on nuclei centroids.
    - Save blended overlay images combining original, labels, contours, centroids, and Voronoi diagrams.
    - Extract patches from Voronoi overlays for downstream analysis.
    """

    # Load pre-trained StarDist nuclei segmentation model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    for tif_path in tif_paths:

        # Prepare output directory paths depending on whether the code is running inside Docker or locally
        basename = tif_path.stem  # Get the filename without extension to use as subfolder name

        if os.path.exists('/.dockerenv'):
            # If running inside Docker, use Docker-specific paths
            output_dir = f'/voronoi_intermediates/{classification}/{basename}'
            final_output_dir = f'/voronoi_tif/{classification}/{basename}'
        else:
            # If running locally, use full local filesystem paths
            output_dir = os.path.join(
                "/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_intermediates",
                classification,
                basename,
            )
            final_output_dir = os.path.join(
                "/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_tif",
                classification,
                basename,
            )

        # Create the directories if they don't already exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(final_output_dir, exist_ok=True)

        # Check and confirm directory creation
        if os.path.exists(output_dir):
            print(f"Directory '{output_dir}' created successfully.")
        else:
            print(f"Failed to create the directory '{output_dir}'.")

        if os.path.exists(final_output_dir):
            print(f"Directory '{final_output_dir}' created successfully.")
        else:
            print(f"Failed to create the directory '{final_output_dir}'.")

        # Load and downsample DAPI channel image
        original_img = load_channel(tif_path, dapi_channel_idx)
        original_img = original_img[::downsample_interval, ::downsample_interval]

        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if original_img.dtype != np.uint8:
            original_img = ((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255).astype(np.uint8)
        original_img = clahe.apply(original_img)

        # Smooth image with Gaussian Blur and morphological closing to reduce noise
        original_img = cv2.GaussianBlur(original_img, (3, 3), 0)
        kernel = np.ones((7, 7), np.uint8)
        original_img = cv2.morphologyEx(original_img, cv2.MORPH_CLOSE, kernel)

        # Predict nuclei masks with StarDist and filter on size constraints
        labelling, _ = model.predict_instances(normalize(original_img), prob_thresh=0.5, nms_thresh=0.3)
        labelling = zoom(labelling, downsample_interval, order=0)
        labelling = filter_on_nuclei_size(labelling, nmin, nmax)

        # Merge adjacent regions by morphological closing and relabel
        kernel = np.ones((5, 5), np.uint8)
        labelling_processed = cv2.morphologyEx(labelling.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        labelling = skimage.measure.label(labelling_processed)

        # Watershed segmentation refinement using distance transform and local maxima markers
        distance = ndimage.distance_transform_edt(labelling > 0)
        local_max_coords = peak_local_max(distance, min_distance=30, labels=labelling > 0)
        local_max_mask = np.zeros(distance.shape, dtype=bool)
        local_max_mask[tuple(local_max_coords.T)] = True
        markers = ndimage.label(local_max_mask)[0]
        refined_labels = watershed(-distance, markers, mask=labelling > 0)
        labelling = refined_labels

        # Skip if no nuclei detected
        if np.sum(labelling) == 0:
            continue

        # Save original and brightened images
        original_img_path = os.path.join(output_dir, f"{basename}_original_image.png")
        cv2.imwrite(original_img_path, original_img)
        print(f"Original image saved at {original_img_path}")

        brightness_increase = 30
        bright_img = cv2.convertScaleAbs(original_img, alpha=2, beta=brightness_increase)
        bright_img_path = os.path.join(output_dir, f"{basename}_brightened_image.png")
        cv2.imwrite(bright_img_path, bright_img)
        print(f"Brightened image saved at {bright_img_path}")

        # Generate and save label visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        labeled_img = label2rgb(labelling, bg_label=0)
        ax.imshow(labeled_img)
        ax.axis('off')
        label_vis_path = os.path.join(output_dir, f"{basename}_labelling_diagram.png")
        plt.savefig(label_vis_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Labeled diagram saved at {label_vis_path}")

        # Extract centroids and contours from labelling
        centroids, contours = process_labelling(labelling, nmin, nmax, original_img)

        # Visualize contours and save
        contour_canvas = np.zeros_like(labelling, dtype=np.uint8)
        for label, contour_list in contours.items():
            cv2.drawContours(contour_canvas, contour_list, -1, (255, 255, 255), thickness=4)
        contours_path = os.path.join(output_dir, f"{basename}_contours_visualization.png")
        cv2.imwrite(contours_path, contour_canvas)

        # Visualize centroids and save
        centroids_canvas = np.zeros_like(labelling, dtype=np.uint8)
        for centroid in np.array(centroids):
            x, y = map(int, centroid)
            if 0 <= x < centroids_canvas.shape[1] and 0 <= y < centroids_canvas.shape[0]:
                cv2.circle(centroids_canvas, (x, y), 5, (255, 255, 255), thickness=-1)
        centroids_path = os.path.join(output_dir, f"{basename}_centroids_visualization.png")
        cv2.imwrite(centroids_path, centroids_canvas)
        print(f"Centroids visualization saved at {centroids_path}")

        # Generate Voronoi diagram from centroids and save
        fig, ax = plt.subplots(figsize=(8, 8))
        vor = Voronoi(np.array(centroids))
        voronoi_plot_2d(vor, ax=ax, show_points=True, show_vertices=True, line_colors="orange", line_width=2)
        ax.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='red', s=30, marker="o")
        img_height, img_width = labelling.shape
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # flip y-axis for image coordinate system
        ax.axis('off')
        voronoi_intermediates_path = os.path.join(output_dir, f"{basename}_voronoi_diagram.png")
        fig.savefig(voronoi_intermediates_path, bbox_inches='tight', dpi=500)
        voronoi_path = os.path.join(final_output_dir, f"{basename}_voronoi_diagram.png")
        fig.savefig(voronoi_path, bbox_inches='tight', dpi=500)

        plt.close(fig)
        print(f"Voronoi diagram saved at {voronoi_path}")

        # Load images for overlaying
        original_image = Image.open(bright_img_path).convert("RGBA")
        label_vis_img = Image.open(label_vis_path).convert("RGBA")
        contours_img = Image.open(contours_path).convert("RGBA")
        centroids_img = Image.open(centroids_path).convert("RGBA")
        voronoi_img = Image.open(voronoi_path).convert("RGBA")

        # Resize label visualization to match original image if needed
        if original_image.size != label_vis_img.size:
            print(f"Resizing label visualization from {label_vis_img.size} to {original_image.size}")
            label_vis_img = label_vis_img.resize(original_image.size)

        # Combine original bright image and label visualization side-by-side and save
        original_labelled_img = np.hstack((np.array(original_image), np.array(label_vis_img)))
        orig_labelled_path = os.path.join(output_dir, f"{basename}_original_labelled_image.png")
        cv2.imwrite(orig_labelled_path, original_labelled_img)

        # Resize contours visualization to match label visualization if needed
        if label_vis_img.size != contours_img.size:
            print(f"Resizing contours visualization from {contours_img.size} to {label_vis_img.size}")
            contours_img = contours_img.resize(label_vis_img.size)

        # Combine label visualization and contours visualization side-by-side and save
        labelled_contours_img = np.hstack((np.array(label_vis_img), np.array(contours_img)))
        labelled_contours_path = os.path.join(output_dir, f"{basename}_labelled_contours_image.png")
        cv2.imwrite(labelled_contours_path, labelled_contours_img)

        # Resize centroids visualization to match original image if needed
        if original_image.size != centroids_img.size:
            print(f"Resizing centroids visualization from {centroids_img.size} to {original_image.size}")
            centroids_img = centroids_img.resize(original_image.size)

        # Change centroid pixels to red color
        centroids_img.putdata([(0, 0, 255, 255) if pixel[0] != 0 else pixel for pixel in centroids_img.getdata()])

        # Blend original image and centroids visualization and save
        original_centroids_img = Image.blend(original_image, centroids_img, alpha=0.4)
        original_centroids_path = os.path.join(output_dir, f"{basename}_original_centroids_image.png")
        original_centroids_img.save(original_centroids_path, format="PNG")

        # Resize centroids visualization to match contours visualization if needed
        if contours_img.size != centroids_img.size:
            print(f"Resizing centroids visualization from {centroids_img.size} to {contours_img.size}")
            centroids_img = centroids_img.resize(contours_img.size)

        # Blend contours and centroids visualizations and save
        contours_centroids_img = Image.blend(contours_img, centroids_img, alpha=0.4)
        contours_centroids_path = os.path.join(output_dir, f"{basename}_contours_centroids_image.png")
        contours_centroids_img.save(contours_centroids_path, format="PNG")

        # Resize Voronoi diagram to match centroids visualization if needed
        if centroids_img.size != voronoi_img.size:
            print(f"Resizing Voronoi diagram from {voronoi_img.size} to {centroids_img.size}")
            voronoi_img = voronoi_img.resize(centroids_img.size)

        # Blend centroids visualization and Voronoi diagram and save
        voronoi_centroids_img = Image.blend(centroids_img, voronoi_img, alpha=0.2)
        voronoi_centroids_path = os.path.join(output_dir, f"{basename}_voronoi_centroids_image.png")
        voronoi_centroids_img.save(voronoi_centroids_path, format="PNG")

        # Resize Voronoi diagram to match original bright image if needed
        if original_image.size != voronoi_img.size:
            print(f"Resizing Voronoi diagram from {voronoi_img.size} to {original_image.size}")
            voronoi_img = voronoi_img.resize(original_image.size)

        # Blend original bright image and Voronoi diagram and save
        # original_voronoi_img = Image.blend(original_image, voronoi_img, alpha=0.3)
        original_voronoi_img = voronoi_img
        original_voronoi_path = os.path.join(output_dir, f"{basename}_original_voronoi_image.png")
        original_voronoi_img.save(original_voronoi_path, format="PNG")

def main():
    """
    Main entry point for nuclei segmentation and Voronoi patch extraction pipeline.

    This function:
    - Parses command-line arguments for nuclei size thresholds, DAPI channel index, and downsampling factor.
    - Sets up input TIFF directories for Cancerous and NotCancerous samples based on the environment (Docker/local).
    - Validates the presence of TIFF files in these directories.
    - Sets up and cleans output directories for Voronoi results and patch extraction.
    - Calls the voronoi processing function on cancerous and non-cancerous TIFF image lists.
    """

    # Parse command-line arguments for segmentation parameters
    parser = argparse.ArgumentParser(description="Segmentation parameters for nuclei detection in TIFF images.")
    parser.add_argument('--nmin', '--nuclei-min-pixels', type=int, required=True,
                        help="Minimum number of pixels for a valid nucleus.")
    parser.add_argument('--nmax', '--nuclei-max-pixels', type=int, required=True,
                        help="Maximum number of pixels for a valid nucleus.")
    parser.add_argument('--didx', '--dapi-channel-idx', type=int, required=True,
                        help="Index of the DAPI channel (typically 0).")
    parser.add_argument('--d', '--downsample-interval', type=int, required=True,
                        help="Factor by which to downsample the image.")
    args = parser.parse_args()

    # Determine TIFF input directories based on environment
    if os.path.exists('/.dockerenv'):
        cancer_tif_dir = '/tif/Cancerous'
        no_cancer_tif_dir = '/tif/NotCancerous'
    else:
        cancer_tif_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif/Cancerous'
        no_cancer_tif_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif/NotCancerous'

    # Verify and list Cancerous TIFF files
    cancer_tif_path_obj = pathlib.Path(cancer_tif_dir)
    if not cancer_tif_path_obj.exists():
        raise FileNotFoundError(f"TIFF directory '{cancer_tif_dir}' does not exist.")
    cancer_tif_paths = list(cancer_tif_path_obj.glob('*.tif'))
    if not cancer_tif_paths:
        raise FileNotFoundError(f"No .tif files found in '{cancer_tif_dir}'.")
    print(f"Cancerous TIFF Paths: {cancer_tif_paths}")

    # Verify and list NotCancerous TIFF files
    no_cancer_tif_path_obj = pathlib.Path(no_cancer_tif_dir)
    if not no_cancer_tif_path_obj.exists():
        raise FileNotFoundError(f"TIFF directory '{no_cancer_tif_dir}' does not exist.")
    no_cancer_tif_paths = list(no_cancer_tif_path_obj.glob('*.tif'))
    if not no_cancer_tif_paths:
        raise FileNotFoundError(f"No .tif files found in '{no_cancer_tif_dir}'.")
    print(f"NotCancerous TIFF Paths: {no_cancer_tif_paths}")

    # Setup Voronoi output directory and clean if exists
    voronoi_intermediates_dir = '/voronoi_intermediates' if os.path.exists('/.dockerenv') else \
        '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_intermediates'
    if os.path.exists(voronoi_intermediates_dir):
        shutil.rmtree(voronoi_intermediates_dir)
        print(f"Directory '{voronoi_intermediates_dir}' has been deleted.")
    os.makedirs(voronoi_intermediates_dir, exist_ok=True)
    print(f"Directory '{voronoi_intermediates_dir}' was created successfully." if os.path.exists(voronoi_intermediates_dir) else
          f"Failed to create the directory '{voronoi_intermediates_dir}'.")

    # Setup Cancerous patch directory
    cancer_patch_intermediates_dir = '/voronoi_intermediates/Cancerous' if os.path.exists('/.dockerenv') else \
        '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_intermediates/Cancerous'
    os.makedirs(cancer_patch_intermediates_dir, exist_ok=True)
    print(f"Directory '{cancer_patch_intermediates_dir}' was created successfully." if os.path.exists(cancer_patch_intermediates_dir) else
          f"Failed to create the directory '{cancer_patch_intermediates_dir}'.")

    # Setup NotCancerous patch directory
    no_cancer_patch_intermediates_dir = '/voronoi_intermediates/NotCancerous' if os.path.exists('/.dockerenv') else \
        '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_intermediates/NotCancerous'
    os.makedirs(no_cancer_patch_intermediates_dir, exist_ok=True)
    print(f"Directory '{no_cancer_patch_intermediates_dir}' was created successfully." if os.path.exists(no_cancer_patch_intermediates_dir) else
          f"Failed to create the directory '{no_cancer_patch_intermediates_dir}'.")

    # Setup segmented patches directory for Cancerous samples, cleaning old directory if present
    cancer_patch_dir = '/voronoi_tif/Cancerous' if os.path.exists('/.dockerenv') else \
        '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_tif/Cancerous'
    if os.path.exists(cancer_patch_dir):
        shutil.rmtree(cancer_patch_dir)
    os.makedirs(cancer_patch_dir, exist_ok=True)
    print(f"Directory '{cancer_patch_dir}' was created successfully." if os.path.exists(cancer_patch_dir) else
          f"Failed to create the directory '{cancer_patch_dir}'.")

    # Setup segmented patches directory for NotCancerous samples, cleaning old directory if present
    no_cancer_patch_dir = '/voronoi_tif/NotCancerous' if os.path.exists('/.dockerenv') else \
        '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_tif/NotCancerous'
    if os.path.exists(no_cancer_patch_dir):
        shutil.rmtree(no_cancer_patch_dir)
    os.makedirs(no_cancer_patch_dir, exist_ok=True)
    print(f"Directory '{no_cancer_patch_dir}' was created successfully." if os.path.exists(no_cancer_patch_dir) else
          f"Failed to create the directory '{no_cancer_patch_dir}'.")

    # Print the number of TIFF images in each category
    print(f"Number of Cancerous TIFFs: {len(cancer_tif_paths)}")
    print(f"Number of NotCancerous TIFFs: {len(no_cancer_tif_paths)}")

    # Run the voronoi processing pipeline for cancerous and non-cancerous images
    voronoi(cancer_tif_paths, "Cancerous", cancer_patch_intermediates_dir, cancer_patch_dir, args.nmin,
            args.nmax, args.didx, args.d)
    voronoi(no_cancer_tif_paths, "NotCancerous", no_cancer_patch_intermediates_dir, no_cancer_patch_dir, args.nmin,
            args.nmax, args.didx, args.d)

if __name__ == "__main__":
    main()
