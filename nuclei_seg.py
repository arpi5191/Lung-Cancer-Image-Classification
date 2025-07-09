# To run:
    # Run File w/o Comparison: python seg.py --nmin 50 --nmax 1000 --didx 0 --d 2
    # Run File w/ Comparison: python seg.py --nmin 500 --nmax 725 --didx 0 --d 2
    # Create image: docker build . -t seg
    # Get a shell into a container: docker run -it -v /Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/seg:/seg seg bash
    # Run Segmentation: docker run -v /Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/seg:/seg -v /Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif:/tif -v /Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/patches:/patches seg
    # Launch virtual environment (https://www.youtube.com/watch?v=WFIZn6titnc):
      # source ~/miniforge3/bin/activate
      # conda install -c apple tensorflow-deps
      # python -m pip uninstall tensoflow-macos
      # python -m pip uninstall tensoflow-metal
      # conda install -c apple tensorflow-deps --force-reinstall
      # source activate moon

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

    # Count the number of pixels for each unique label (nucleus)
    # Index 0 corresponds to background; higher indices correspond to labeled nuclei
    segmented_cell_sizes = np.bincount(labeling.ravel())

    # Create boolean arrays marking which nuclei are too small or too large
    too_small = segmented_cell_sizes < nmin
    too_large = segmented_cell_sizes > nmax

    # Combine the two conditions to identify all invalid nuclei
    too_small_or_large = too_small | too_large

    # Zero out all pixels in the labeling array that belong to invalid nuclei
    # This works by using the original labeling array as an index into the boolean mask
    labeling[too_small_or_large[labeling]] = 0

    # Return the cleaned labeling array
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

    Returns:
    - patch_dims: Dictionary mapping each image basename to its extracted patch data
    """

    # Load the pre-trained StarDist model for general fluorescence nuclei segmentation
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # Dictionary to store patch-related information for each image
    patch_dims = {}

    # Track the total number of valid nuclei patches processed
    num_patches = 0

    # Loop through each TIFF image path
    for tif_path in tif_paths:

        # Get the file name without the .tif extension (used for naming outputs)
        basename = tif_path.name.rstrip('.tif')

        # Load only the DAPI channel and downsample by skipping pixels (e.g., every 4th pixel)
        img = load_channel(tif_path, dapi_channel_idx)
        img = img[::downsample_interval, ::downsample_interval]

        # Predict labeled nuclei masks from the normalized image using StarDist
        labeling, _ = model.predict_instances(normalize(img))

        # Resize labeling back to original image resolution using nearest-neighbor interpolation
        labeling = zoom(labeling, 1, order=0)
        # labeling = zoom(labeling, downsample_interval, order=0)  # optional alternative

        # Remove very small or very large nuclei
        labeling = filter_on_nuclei_size(labeling, nmin, nmax)

        # Check if segmentation produced any nuclei at all
        if np.sum(labeling) == 0:
            print(f"Segmentation failed for {basename}")

        # Process the labeled output to extract valid centroids of nuclei
        # This also filters out bad contours internally
        output_dir, sorted_labels, centroids = process_labeling(labeling, nmin, nmax, img, classification, basename)

        # NOTE: 'patches' is not defined in this scope. You likely intended to collect patches during process_labeling.
        # You should modify process_labeling to return patches if needed, or remove this line:
        # patch_dims[basename] = patches
        # For now, we'll set patch_dims[basename] = centroids (or remove entirely if unused)
        centroid_clusters = merge_focus_with_closest_neighbors(img, classification, basename, output_dir, sorted_labels, centroids, num_other_centroids)

        patch_dims[basename] = centroids  # ← temporary fix until patches are actually collected

        # Update total count of valid nuclei patches
        num_patches += len(centroid_clusters)

        # Save the labeled segmentation mask as a NumPy file for future use
        print(f'Output Path: {seg_dir}/{basename}_seg.npy')
        np.save(f'{seg_dir}/{basename}_seg.npy', labeling)

        # Print a blank line for readability between image logs
        print()

    # Summary printout after all files processed
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

    # Compute the convex hull of the contour: the smallest convex polygon enclosing the shape
    hull = cv2.convexHull(contour)

    # Compute the area of the convex hull
    hull_area = cv2.contourArea(hull)

    # Calculate solidity as the ratio of contour area to convex hull area
    # If the hull area is zero (e.g., degenerate shape), return 0 to avoid division by zero
    solidity = contour_area / hull_area if hull_area > 0 else 0

    return solidity

def process_labeling(labeling, nmin, nmax, img, classification, basename, minimum_solidity=0.4):
    """
    Process labeled nuclei to:
    - Filter out invalid labels based on area and solidity thresholds
    - Compute centroid coordinates for valid nuclei

    Parameters:
    - labeling: 2D numpy array where each nucleus is represented by a unique integer label
    - nmin: Minimum valid nucleus area (in pixels)
    - nmax: Maximum valid nucleus area (in pixels)
    - img: Original image (not used in this function but typically passed for patch extraction)
    - classification: Subfolder name used for organizing output
    - basename: Base name of the image/file used in output folder naming
    - minimum_solidity: Minimum allowed solidity (0 to 1) for contour acceptance

    Returns:
    - output_dir: Path where patches will be saved
    - sorted_labels: List of valid nucleus labels sorted ascending
    - centroids: Dictionary mapping label -> (x, y) centroid coordinates for valid nuclei
    """

    # Determine output directory based on environment (Docker vs local)
    if os.path.exists('/.dockerenv'):
        output_dir = f'/patches/{classification}/{basename}'  # Inside Docker container
    else:
        output_dir = f'/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/patches/{classification}/{basename}'  # Local environment

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Confirm directory creation
    if os.path.exists(output_dir):
        print(f"Directory '{output_dir}' created successfully.")
    else:
        print(f"Failed to create the directory '{output_dir}'.")

    # Dictionary to hold centroid coordinates of valid nuclei
    centroids = {}

    # Get all unique labels from the labeling array (each corresponds to one segmented nucleus)
    unique_labels = np.unique(labeling)

    # Iterate over each label to validate and extract centroid info
    for label in unique_labels:
        if label == 0:
            # Label 0 is background; skip
            continue

        # Create a binary mask for the current label
        binary_mask = np.where(labeling == label, 1, 0).astype(np.uint8)

        # Find contours of the current nucleus; retrieve only external contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out labels that do not have exactly one contour (removes noise or oversegmentation)
        if len(contours) != 1:
            labeling[labeling == label] = 0  # Remove this label from labeling
            continue

        # Use the single contour found to compute area
        contour = contours[0]
        area = cv2.contourArea(contour)

        # Calculate solidity (ratio of contour area to convex hull area)
        solidity = compute_solidity(contour, area)

        # Reject nuclei based on size thresholds or solidity or if contour is empty
        if area < nmin or area > nmax or solidity < minimum_solidity or len(contour) == 0:
            labeling[labeling == label] = 0  # Remove invalid label
            continue

        # Compute centroid coordinates using image moments
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0  # Degenerate contour fallback

        # Save centroid for this label
        centroids[label] = [cX, cY]

    # Sort the valid labels in ascending order
    sorted_labels = sorted(centroids.keys())

    return output_dir, sorted_labels, centroids

def merge_focus_with_closest_neighbors(img, classification, basename, output_dir, sorted_labels, centroids, num_other_centroids):
    """
    For each nucleus (focus nucleus), find its closest neighboring nuclei based on centroid distances,
    form a merged cluster of centroids (focus + neighbors), and extract patches from the image.

    Parameters:
    - img: Image array from which patches are extracted.
    - classification: String label/classification for the nuclei.
    - basename: Base filename for saving outputs.
    - output_dir: Directory to save patches.
    - sorted_labels: List of nucleus labels (e.g., sorted by size or other criteria).
    - centroids: Dictionary mapping label -> (x, y) centroid coordinates.
    - num_other_centroids: Number of closest neighboring nuclei to include per focus nucleus.

    Returns:
    - centroid_clusters: Dictionary mapping each focus nucleus label to a list of its closest neighbor labels.
    """

    centroid_clusters = {}   # Maps each focus nucleus to its closest neighbor labels
    merged_contours = {}     # Stores merged centroid points for each cluster
    seen_clusters = set()    # Tracks unique clusters to avoid redundant patch generation

    # Iterate through all nuclei as focus nuclei
    for i in range(len(sorted_labels)):
        label1 = sorted_labels[i]
        cX1, cY1 = centroids[label1]

        distances = {}

        # Compute distances from current nucleus to all others
        for j in range(len(sorted_labels)):
            if i != j:
                label2 = sorted_labels[j]
                cX2, cY2 = centroids[label2]
                distance = math.dist([cX1, cY1], [cX2, cY2])
                distances[label2] = distance

        # Select the N nearest neighbor nuclei
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        closest_labels = [label for label, _ in sorted_distances[:num_other_centroids]]

        # Create a sorted, hashable tuple of this nucleus cluster
        cluster_key = tuple(sorted([label1] + closest_labels))

        # Before adding a new cluster_key:
        if any(differs_by_four_or_less(cluster_key, existing) for existing in seen_clusters):
            # Skip adding this cluster because it’s too similar (differs by 0 or 1 label)
            continue
        seen_clusters.add(cluster_key)

        # Save this cluster's labels for later tracking
        centroid_clusters[label1] = closest_labels

        # Gather centroid coordinates for patch extraction
        combined_points = [centroids[label] for label in cluster_key]
        merged_contours[label1] = np.vstack(combined_points)

        # Extract and save patch from combined cluster region
        get_patches(merged_contours[label1], img, label1, basename, output_dir)

    return centroid_clusters

def differs_by_four_or_less(set1, set2):
    """
    Check if two clusters differ by at most one element.

    Converts the inputs to sets and compares their symmetric difference.
    Returns True if they differ by zero or one element, else False.
    """
    set1 = set(set1)
    set2 = set(set2)
    return len(set1.symmetric_difference(set2)) <= 4

def get_patches(region, img, label, basename, output_dir, padding=50, patch_width=512, patch_height=512):
    """
    Extracts and saves an image patch around specified centroid points with padding and resizing.

    Parameters:
    - region (np.ndarray): Array of centroid points (x, y) defining the region of interest.
    - img (np.ndarray): Original image array from which to extract patches.
    - label (int or str): Label identifier for the nucleus or cluster, used for naming the patch file.
    - basename (str): Base filename or identifier used in the patch filename.
    - output_dir (str): Directory path where the extracted patch will be saved.
    - padding (int, optional): Number of pixels to pad around the bounding box of centroid points. Default is 50.
    - patch_width (int, optional): Width in pixels to resize the extracted patch. Default is 512.
    - patch_height (int, optional): Height in pixels to resize the extracted patch. Default is 512.

    Process:
    - Calculates a bounding box around the provided centroid points, extending it by the padding.
    - Clips bounding box coordinates to image boundaries to avoid out-of-range errors.
    - Crops the patch from the original image using the bounding box.
    - Resizes the cropped patch to the specified width and height using linear interpolation.
    - Applies contrast scaling to enhance the patch visibility.
    - Saves the processed patch as a TIFF image with a filename incorporating the basename and label.

    Returns:
    - None (patch is saved directly to disk).
    """

    # Get the height and width of the original image
    image_height, image_width = img.shape[:2]

    # Extract x and y coordinates from the region points (centroids)
    x_points = region[:, 0]
    y_points = region[:, 1]

    # Calculate bounding box coordinates with padding around the region points
    # Ensure coordinates don't go out of image bounds by clipping with max and min
    minX = max(0, min(x_points) - padding)
    minY = max(0, min(y_points) - padding)
    maxX = min(image_width, max(x_points) + padding)
    maxY = min(image_height, max(y_points) + padding)

    # Crop the patch from the original image using the bounding box coordinates
    patch = img[minY:maxY, minX:maxX]

    # Resize the extracted patch to the desired fixed size (e.g., 512x512)
    # Using linear interpolation for resizing
    patch = cv2.resize(patch, (patch_width, patch_height), interpolation=cv2.INTER_LINEAR)

    # Create a filename for the patch including the basename and label
    patch_filename = f'{basename}_label{label}.tif'

    # Apply contrast scaling to the patch for better visibility (alpha=10.0 scales pixel intensities)
    patch = cv2.convertScaleAbs(patch, alpha=10.0, beta=0)

    # Construct the full output file path
    patch_output_path = os.path.join(output_dir, patch_filename)

    # Save the patch image to disk
    cv2.imwrite(patch_output_path, patch)

def main():

    # Define and parse the command-line arguments for input parameters
    parser = argparse.ArgumentParser(description="Segmentation parameters for nuclei detection in TIFF images.")
    parser.add_argument('--nmin', '--nuclei-min-pixels', type=int, help="Minimum number of pixels for a valid nucleus.")
    parser.add_argument('--nmax', '--nuclei-max-pixels', type=int, help="Maximum number of pixels for a valid nucleus.")
    parser.add_argument('--didx', '--dapi-channel-idx', type=int, help="Index of the DAPI channel (typically 0).")
    parser.add_argument('--d', '--downsample-interval', type=int, help="Factor by which to downsample the image.")
    args = parser.parse_args()

    global seg_dir

    # Determine output segmentation directory based on the environment (Docker or local)
    if os.path.exists('/.dockerenv'):
        seg_dir = '/seg'  # Docker output directory
    else:
        seg_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/seg'  # Local output directory

    # Check if the output segmentation directory already exists and delete if necessary
    if os.path.exists(seg_dir):
        shutil.rmtree(seg_dir)
        print(f"Directory '{seg_dir}' has been deleted.")
    else:
        print(f"Directory '{seg_dir}' does not exist.")

    # Create the output segmentation directory
    os.makedirs(seg_dir, exist_ok=True)

    # Confirm directory creation
    if os.path.exists(seg_dir):
        print(f"Directory '{seg_dir}' created successfully.")
    else:
        print(f"Failed to create the directory '{seg_dir}'.")

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
        patch_dir = '/patches'  # Output path for Docker environment
    else:
        patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/patches'  # Output path for local environment

    # Check if the output directory already exists
    if os.path.exists(patch_dir):
        # Remove the existing directory and all its contents
        shutil.rmtree(patch_dir)
        print(f"Directory '{patch_dir}' has been deleted.")
    else:
        print(f"Directory '{patch_dir}' does not exist.")

    # Create the output directory if it doesn't exist
    os.makedirs(patch_dir, exist_ok=True)

    # Confirm the creation of the output directory and print the status
    if os.path.exists(patch_dir):
        print(f"Directory '{patch_dir}' was created successfully.")
    else:
        print(f"Failed to create the directory '{patch_dir}'.")

    # Set the output directory for patches of cancerous images based on the environment (Docker or local)
    if os.path.exists('/.dockerenv'):
        cancer_patch_dir = '/patches/Cancerous'  # Output path for Docker environment
    else:
        cancer_patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/patches/Cancerous'  # Output path for local environment

    # Create the cancerous patch directory if it doesn't already exist
    os.makedirs(cancer_patch_dir, exist_ok=True)

    # Verify the creation of the cancerous patch directory and print its status
    if os.path.exists(cancer_patch_dir):
        print(f"Directory '{cancer_patch_dir}' was created successfully.")
    else:
        print(f"Failed to create the directory '{cancer_patch_dir}'.")

    # Set the output directory for patches of non-cancerous images based on the environment (Docker or local)
    if os.path.exists('/.dockerenv'):
        no_cancer_patch_dir = '/patches/NotCancerous'  # Output path for Docker environment
    else:
        no_cancer_patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/patches/NotCancerous'  # Output path for local environment

    # Create the non-cancerous patch directory if it doesn't already exist
    os.makedirs(no_cancer_patch_dir, exist_ok=True)

    # Verify the creation of the non-cancerous patch directory and print its status
    if os.path.exists(no_cancer_patch_dir):
        print(f"Directory '{no_cancer_patch_dir}' was created successfully.")
    else:
        print(f"Failed to create the directory '{no_cancer_patch_dir}'.")

    # Perform nuclei segmentation on loaded TIFF images in the cancerous and non-cancerous datasets
    patch_dims = segment_images(cancer_tif_paths, "Cancerous", args.nmin, args.nmax, args.didx, args.d)
    patch_dims = segment_images(no_cancer_tif_paths, "NotCancerous", args.nmin, args.nmax, args.didx, args.d)

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------------------------------------------

## Future Considerations
    # Implement parallel processing to speed up segmentation across multiple TIFF paths
    # Develop a method to precisely locate and annotate nuclei positions within the images

# 643 cancerous patches
# 171 non-cancerous patches
