# Import packages
import os
import cv2
import math
import time
import shutil
import pathlib
import tifffile
import argparse
import numpy as np
from scipy.ndimage import zoom
from csbdeep.utils import normalize
import skimage.exposure as exposure
from stardist.models import StarDist2D

def load_channel(tif_path, channel_idx):
    '''
    Load the specified channel from a multi-channel TIFF file as a 2D numpy array.

    Args:
        - tif_path: str, the file path to the multi-channel TIFF image.
        - channel_idx: int, the 0-based index of the desired channel (e.g., DAPI is often at index 0).

    Returns:
        - img: 2D numpy array, the selected channel of the image with intensity rescaled to 8-bit.
    '''

    # Read the specified channel from the TIFF file
    channel = tifffile.imread(tif_path)

    # Rescale intensity values from original dynamic range to [0, 255] (8-bit)
    return exposure.rescale_intensity(channel, in_range='image', out_range=(0, 255)).astype(np.uint8)


def filter_on_nuclei_size(labeling, nmin, nmax):
    """
    Filters labeled nuclei based on size (pixel area).
    Removes nuclei that are too small or too large by setting their labels to 0.

    Parameters:
    - labeling: 2D numpy array where each labeled region represents a nucleus
    - nmin: Minimum allowed nucleus size (in pixels)
    - nmax: Maximum allowed nucleus size (in pixels)

    Returns:
    - labeling: Modified array with small and large nuclei removed (set to label 0)
    """

    # Count number of pixels in each labeled region (label=0 is background)
    segmented_cell_sizes = np.bincount(labeling.ravel())

    # Create masks for labels that are too small or too large
    too_small = segmented_cell_sizes < nmin
    too_large = segmented_cell_sizes > nmax

    # Combine both masks
    too_small_or_large = too_small | too_large

    # Set the labels of too-small or too-large regions to 0 (background)
    labeling[too_small_or_large[labeling]] = 0

    return labeling

def segment_images(tif_paths, classification, nmin, nmax, dapi_channel_idx, downsample_interval, num_other_centroids=4):
    """
    Segments nuclei in a list of TIFF images using a pre-trained StarDist model.

    Parameters:
    - tif_paths: List of Path objects for the input .tif images
    - classification: String label used for organizing output directories
    - nmin: Minimum pixel area for valid nuclei
    - nmax: Maximum pixel area for valid nuclei
    - dapi_channel_idx: Index of the DAPI channel in multi-channel images
    - downsample_interval: Factor by which to downsample the image
    - num_other_centroids: Number of nearby nuclei to consider per focus nucleus when creating patches

    Returns:
    - patch_dims: Dictionary mapping each image basename to its extracted centroid info
    """

    # Load the pre-trained StarDist model for fluorescence nuclei
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    patch_dims = {}
    num_patches = 0

    # Loop over each input TIFF file
    for tif_path in tif_paths:
        basename = tif_path.name.rstrip('.tif')

        # Load DAPI channel and downsample spatially
        img = load_channel(tif_path, dapi_channel_idx)
        img = img[::downsample_interval, ::downsample_interval]

        # Normalize and segment using StarDist
        labeling, _ = model.predict_instances(normalize(img))

        # Upsample labeling back to original resolution (or keep unchanged)
        labeling = zoom(labeling, 1, order=0)
        # labeling = zoom(labeling, downsample_interval, order=0)  # optional version

        # Filter out too-small or too-large segmented nuclei
        labeling = filter_on_nuclei_size(labeling, nmin, nmax)

        if np.sum(labeling) == 0:
            print(f"Segmentation failed for {basename}")

        # Identify valid nuclei and their centroids
        output_dir, sorted_labels, centroids = process_labeling(
            labeling, nmin, nmax, img, classification, basename
        )

        # Find closest clusters of nuclei and extract patches
        centroid_clusters = merge_focus_with_closest_neighbors(
            img, classification, basename, output_dir,
            sorted_labels, centroids, num_other_centroids
        )

        patch_dims[basename] = centroids  # Could be replaced with actual patches

        num_patches += len(centroid_clusters)

        # Save segmentation mask for future inspection
        print(f'Output Path: {seg_dir}/{basename}_seg.npy')
        np.save(f'{seg_dir}/{basename}_seg.npy', labeling)
        print()

    # Print summary
    print("Classification:", classification)
    print("Number of Patches:", num_patches)
    print()

    return patch_dims

def compute_solidity(contour, contour_area):
    """
    Computes the solidity of a contour.

    Solidity is defined as the ratio of the contour area to the area of its convex hull.
    It gives a measure of how "solid" or "compact" a shape is. A value close to 1 means the shape
    is relatively convex, while lower values indicate concavity or irregularity.

    Parameters:
    - contour: The contour points of the shape (as returned by cv2.findContours)
    - contour_area: Precomputed area of the contour (in pixels)

    Returns:
    - solidity: A float value in [0, 1] representing the compactness of the shape
    """

    # Get the convex hull around the contour
    hull = cv2.convexHull(contour)

    # Compute the area of the convex hull
    hull_area = cv2.contourArea(hull)

    # Return solidity as ratio of contour to hull area
    return contour_area / hull_area if hull_area > 0 else 0

def process_labeling(labeling, nmin, nmax, img, classification, basename, minimum_solidity=0.4):
    """
    Process labeled nuclei to:
    - Filter out invalid labels based on area and solidity thresholds
    - Compute centroid coordinates for valid nuclei

    Parameters:
    - labeling: 2D numpy array where each nucleus is represented by a unique integer label
    - nmin: Minimum valid nucleus area (in pixels)
    - nmax: Maximum valid nucleus area (in pixels)
    - img: Original image (not used here, passed for compatibility or future use)
    - classification: Subfolder name used for organizing output
    - basename: Base name of the image/file used in output folder naming
    - minimum_solidity: Minimum allowed solidity (0 to 1) for contour acceptance

    Returns:
    - output_dir: Path where patches will be saved
    - sorted_labels: List of valid nucleus labels, sorted in ascending order
    - centroids: Dictionary mapping label -> [x, y] centroid coordinates for valid nuclei
    """

    # Set output directory based on environment
    if os.path.exists('/.dockerenv'):
        # If running inside a Docker container
        output_dir = f'/nuclei_patches/{classification}/{basename}'
    else:
        # If running locally on Arpitha's machine
        output_dir = f'/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/nuclei_patches/{classification}/{basename}'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Print confirmation of directory creation
    if os.path.exists(output_dir):
        print(f"Directory '{output_dir}' created successfully.")
    else:
        print(f"Failed to create the directory '{output_dir}'.")

    # Dictionary to store centroids of valid nuclei
    centroids = {}

    # Get all unique labels in the segmentation mask
    unique_labels = np.unique(labeling)

    # Iterate over each label to validate it and compute centroid
    for label in unique_labels:
        if label == 0:
            continue  # Skip background label

        # Create a binary mask for this label
        binary_mask = np.where(labeling == label, 1, 0).astype(np.uint8)

        # Extract external contours of the object
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Skip labels with more than one contour (indicates oversegmentation or noise)
        if len(contours) != 1:
            labeling[labeling == label] = 0
            continue

        # Get the only contour
        contour = contours[0]

        # Compute area of the contour
        area = cv2.contourArea(contour)

        # Compute solidity: area / convex hull area
        solidity = compute_solidity(contour, area)

        # Reject nucleus if it fails any criteria (size, solidity, empty contour)
        if area < nmin or area > nmax or solidity < minimum_solidity or len(contour) == 0:
            labeling[labeling == label] = 0
            continue

        # Compute centroid using image moments
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # Fallback if contour is degenerate
            cX, cY = 0, 0

        # Store centroid for valid label
        centroids[label] = [cX, cY]

    # Sort valid labels for downstream reproducibility
    sorted_labels = sorted(centroids.keys())

    return output_dir, sorted_labels, centroids

def merge_focus_with_closest_neighbors(img, classification, basename, output_dir, sorted_labels, centroids, num_other_centroids):
    """
    For each nucleus (focus nucleus), find its N closest neighbors based on Euclidean distance between centroids.
    Form a unique cluster of centroids (focus + neighbors) and extract patches from the image.

    Parameters:
    - img: 2D NumPy array of the original image (used for patch extraction).
    - classification: Classification label used in output folder structure.
    - basename: Base filename of the image used for naming outputs.
    - output_dir: Directory to save the extracted patches.
    - sorted_labels: List of nucleus labels, typically sorted.
    - centroids: Dictionary mapping label -> (x, y) centroid coordinates.
    - num_other_centroids: Number of nearest neighbors to include with each focus nucleus.

    Returns:
    - centroid_clusters: Dictionary mapping each focus nucleus label to a list of its closest neighbor labels.
    """

    centroid_clusters = {}   # Maps each nucleus to its nearest neighbor cluster
    merged_contours = {}     # Stores combined centroid coordinates for patch extraction
    seen_clusters = set()    # Avoids generating patches for similar clusters repeatedly

    # Loop over each label to treat as a focus nucleus
    for i in range(len(sorted_labels)):
        label1 = sorted_labels[i]
        cX1, cY1 = centroids[label1]

        distances = {}

        # Compute distance from the current nucleus to all other nuclei
        for j in range(len(sorted_labels)):
            if i != j:
                label2 = sorted_labels[j]
                cX2, cY2 = centroids[label2]
                distance = math.dist([cX1, cY1], [cX2, cY2])
                distances[label2] = distance

        # Sort other nuclei by increasing distance and select N closest
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        closest_labels = [label for label, _ in sorted_distances[:num_other_centroids]]

        # Define a cluster as a sorted tuple of focus + neighbors (for uniqueness checking)
        cluster_key = tuple(sorted([label1] + closest_labels))

        # Skip clusters too similar to any previously seen cluster (by ≤ 4 label difference)
        if any(differs_by_four_or_less(cluster_key, existing) for existing in seen_clusters):
            continue
        seen_clusters.add(cluster_key)

        # Save the neighbor labels for this focus nucleus
        centroid_clusters[label1] = closest_labels

        # Stack the (x, y) points of all nuclei in the cluster into a single array
        combined_points = [centroids[label] for label in cluster_key]
        merged_contours[label1] = np.vstack(combined_points)

        # Extract and save a patch covering all nuclei in this cluster
        get_patches(merged_contours[label1], img, label1, basename, output_dir)

    return centroid_clusters

def differs_by_four_or_less(set1, set2):
    """
    Check if two clusters differ by at most four elements.

    Converts the inputs to sets and compares their symmetric difference.
    Returns True if they differ by four or fewer elements, else False.
    """
    set1 = set(set1)
    set2 = set(set2)
    return len(set1.symmetric_difference(set2)) <= 4

def get_patches(region, img, label, basename, output_dir, padding=50, patch_width=512, patch_height=512):
    """
    Extract and save a rectangular image patch centered around a group of centroid points.

    Parameters:
    - region (np.ndarray): Array of (x, y) centroid coordinates defining the region of interest.
    - img (np.ndarray): The original grayscale image from which to extract the patch.
    - label (int or str): Identifier used in the output filename to track the nucleus or cluster.
    - basename (str): Base name of the image file, used in naming the output patch.
    - output_dir (str): Path to the directory where the patch image will be saved.
    - padding (int, optional): Number of pixels to extend beyond the bounding box of the region. Default is 50.
    - patch_width (int, optional): Desired output patch width (after resizing). Default is 512.
    - patch_height (int, optional): Desired output patch height (after resizing). Default is 512.

    Steps:
    - Calculate the bounding box around the input region with additional padding.
    - Clip the bounding box to image boundaries.
    - Crop the region from the original image.
    - Resize to fixed dimensions and apply contrast enhancement.
    - Save the processed patch as a .tif file.

    Returns:
    - None. The patch is saved directly to disk.
    """

    # Get image dimensions
    image_height, image_width = img.shape[:2]

    # Split region points into x and y coordinate arrays
    x_points = region[:, 0]
    y_points = region[:, 1]

    # Compute the bounding box with padding, ensuring it stays within image bounds
    minX = max(0, min(x_points) - padding)
    minY = max(0, min(y_points) - padding)
    maxX = min(image_width, max(x_points) + padding)
    maxY = min(image_height, max(y_points) + padding)

    # Crop the image patch from the original image using bounding box coordinates
    patch = img[minY:maxY, minX:maxX]

    # Resize the patch to the desired output size (512x512 by default)
    patch = cv2.resize(patch, (patch_width, patch_height), interpolation=cv2.INTER_LINEAR)

    # Enhance contrast: scale pixel intensities (alpha=10 makes details more visible)
    patch = cv2.convertScaleAbs(patch, alpha=10.0, beta=0)

    # Create the filename for this patch
    patch_filename = f'{basename}_label{label}.tif'

    # Combine directory and filename to get full save path
    patch_output_path = os.path.join(output_dir, patch_filename)

    # Save the patch image to disk
    cv2.imwrite(patch_output_path, patch)

def main():
    """
    Main function to perform nuclei segmentation on multi-channel TIFF images.

    Workflow:
    - Parses command-line arguments for segmentation parameters.
    - Sets up directory paths based on execution environment (Docker or local).
    - Clears and recreates output directories for segmentation masks and nuclei patches.
    - Loads TIFF image paths for Cancerous and NotCancerous categories.
    - Runs segmentation on both categories using the specified parameters.
    """

    # Parse command-line arguments for segmentation parameters
    parser = argparse.ArgumentParser(description="Segmentation parameters for nuclei detection in TIFF images.")
    parser.add_argument('--nmin', '--nuclei-min-pixels', type=int, help="Minimum number of pixels for a valid nucleus.")
    parser.add_argument('--nmax', '--nuclei-max-pixels', type=int, help="Maximum number of pixels for a valid nucleus.")
    parser.add_argument('--didx', '--dapi-channel-idx', type=int, help="Index of the DAPI channel (typically 0).")
    parser.add_argument('--d', '--downsample-interval', type=int, help="Factor by which to downsample the image.")
    args = parser.parse_args()

    global seg_dir

    # Set segmentation output directory depending on environment (Docker vs Local)
    if os.path.exists('/.dockerenv'):
        seg_dir = '/nuclei_seg'  # Docker container path
    else:
        seg_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/nuclei_seg'  # Local path

    # Remove existing segmentation directory to start fresh
    if os.path.exists(seg_dir):
        shutil.rmtree(seg_dir)
        print(f"Directory '{seg_dir}' has been deleted.")
    else:
        print(f"Directory '{seg_dir}' does not exist.")

    # Create segmentation directory
    os.makedirs(seg_dir, exist_ok=True)
    print(f"Directory '{seg_dir}' created successfully." if os.path.exists(seg_dir) else f"Failed to create the directory '{seg_dir}'.")

    # Set input TIFF directories for cancerous and non-cancerous images
    if os.path.exists('/.dockerenv'):
        cancer_tif_dir = '/tif/Cancerous'
        no_cancer_tif_dir = '/tif/NotCancerous'
    else:
        cancer_tif_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif/Cancerous'
        no_cancer_tif_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif/NotCancerous'

    # Verify existence of cancerous TIFF directory and list TIFF files
    if not pathlib.Path(cancer_tif_dir).exists():
        raise FileNotFoundError(f"TIFF directory '{cancer_tif_dir}' does not exist. Please check the path.")
    cancer_tif_paths = list(pathlib.Path(cancer_tif_dir).glob('*.tif'))
    print(f"Cancerous TIFF Paths: {cancer_tif_paths}")
    if not cancer_tif_paths:
        raise FileNotFoundError(f"The directory '{cancer_tif_dir}' contains no .tif files.")

    # Verify existence of non-cancerous TIFF directory and list TIFF files
    if not pathlib.Path(no_cancer_tif_dir).exists():
        raise FileNotFoundError(f"TIFF directory '{no_cancer_tif_dir}' does not exist. Please check the path.")
    no_cancer_tif_paths = list(pathlib.Path(no_cancer_tif_dir).glob('*.tif'))
    print(f"NotCancerous TIFF Paths: {no_cancer_tif_paths}")
    if not no_cancer_tif_paths:
        raise FileNotFoundError(f"The directory '{no_cancer_tif_dir}' contains no .tif files.")

    # Setup output directories for nuclei patches depending on environment
    if os.path.exists('/.dockerenv'):
        patch_dir = '/nuclei_patches'  # Docker path
    else:
        patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/nuclei_patches'  # Local path

    # Remove existing patches directory if present
    if os.path.exists(patch_dir):
        shutil.rmtree(patch_dir)
        print(f"Directory '{patch_dir}' has been deleted.")
    else:
        print(f"Directory '{patch_dir}' does not exist.")

    # Create patches directory
    os.makedirs(patch_dir, exist_ok=True)
    print(f"Directory '{patch_dir}' was created successfully." if os.path.exists(patch_dir) else f"Failed to create the directory '{patch_dir}'.")

    # Setup cancerous patches directory
    if os.path.exists('/.dockerenv'):
        cancer_patch_dir = '/nuclei_patches/Cancerous'
    else:
        cancer_patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/nuclei_patches/Cancerous'
    os.makedirs(cancer_patch_dir, exist_ok=True)
    print(f"Directory '{cancer_patch_dir}' was created successfully." if os.path.exists(cancer_patch_dir) else f"Failed to create the directory '{cancer_patch_dir}'.")

    # Setup non-cancerous patches directory
    if os.path.exists('/.dockerenv'):
        no_cancer_patch_dir = '/nuclei_patches/NotCancerous'
    else:
        no_cancer_patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/nuclei_patches/NotCancerous'
    os.makedirs(no_cancer_patch_dir, exist_ok=True)
    print(f"Directory '{no_cancer_patch_dir}' was created successfully." if os.path.exists(no_cancer_patch_dir) else f"Failed to create the directory '{no_cancer_patch_dir}'.")

    # Run segmentation on cancerous TIFF images
    patch_dims = segment_images(cancer_tif_paths, "Cancerous", args.nmin, args.nmax, args.didx, args.d)

    # Run segmentation on non-cancerous TIFF images
    patch_dims = segment_images(no_cancer_tif_paths, "NotCancerous", args.nmin, args.nmax, args.didx, args.d)


if __name__ == "__main__":
    main()
