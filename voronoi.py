# To run:
    # Run File: python voronoi.py --nmin 50 --nmax 1000 --didx 0 --d 2
    # Create image: docker build . -t voronoi
    # Get a shell into a container: docker run -it -v /Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi:/voronoi voronoi bash
    # Run Segmentation: docker run -v /Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi:/voronoi -v /Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_tif:/voronoi_tif -v /Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi:/voronoi voronoi_tif
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

def segment_images(tif_paths, classification, nmin, nmax, dapi_channel_idx, downsample_interval):

    # Load the pre-trained StarDist model for nuclei detection
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # Create a dictionary to store patch dimensions
    patch_dims = {}

    # Iterate through each .tif file path
    for tif_path in tif_paths:

        # Extract the base name of the file (without extension) for labeling purposes
        basename = tif_path.name.rstrip('.tif')
        # print("Segmenting tif path: {}".format(basename))

        # Load and process the DAPI channel (resize/modify resolution)
        # Downsample the image by taking every nth pixel, as specified by downsample_interval
        img = load_channel(tif_path, dapi_channel_idx)
        img = img[::downsample_interval, ::downsample_interval]

        # Predict nuclei instances using the StarDist model
        # Downsample the labeling array using nearest-neighbor interpolation
        # Filter nuclei based on size, keeping only those within the specified pixel range (nmin to nmax)
        labeling, _ = model.predict_instances(normalize(img))
        labeling = zoom(labeling, 1, order=0)
        # labeling = zoom(labeling, downsample_interval, order=0)
        labeling = filter_on_nuclei_size(labeling, nmin, nmax)

        # Check if segmentation was successful by evaluating the sum of the labeled areas
        if sum(sum(labeling)) == 0:
            print(f"Segmentation failed for {basename}")

        # Process the labeled nuclei to extract contours and patch information
        filtered_contours, size_contours, patches = process_labeling(labeling, nmin, nmax, img, classification, basename)

        # Store the patches associated with the current basename in the patch dimensions dictionary
        # Save patch images to the output directory for future processing
        patch_dims[basename] = patches

        # # Print the sizes of the filtered contours and the details of the patches
        # print(f"Contours Sizes: {size_contours}")
        # print(f"Patches: {patches}")

        # Save the labeled output to a .npy file in the specified output directory
        print(f'Output Path: {seg_dir}/{basename}_seg.npy')
        np.save(f'{seg_dir}/{basename}_seg.npy', labeling)

        # Add a space between outputs for clarity
        print()

    # Return the patches result
    return patch_dims

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

def process_labeling(labeling, nmin, nmax, img, classification, basename, minimum_solidity = 0.8):

    # Check if the script is running in a Docker environment
    if os.path.exists('/.dockerenv'):
        output_dir = f'/patches/{classification}/{basename}'  # Path for Docker
    else:
        output_dir = f'/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/patches/{classification}/{basename}'  # Path for local

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Confirm if the directory was created successfully
    if os.path.exists(output_dir):
        print(f"Directory '{output_dir}' created successfully.")
    else:
        print(f"Failed to create the directory '{output_dir}'.")

    # Initialize dictionaries to store contour sizes, filtered contours, and patches for each nuclei label
    size_contours = {}
    filtered_contours = {}
    patches = {}

    # Get unique labels from the labeling array
    unique_labels = np.unique(labeling)

    # Iterate through each unique label
    for label in unique_labels:
         # Ignore the background label (0)
        if label != 0:
            # Print the label
            # print("Label: {}".format(label))

            # Create a binary mask for the current label
            binary_mask = np.where(labeling == label, 1, 0).astype(np.uint8)

            # Find contours in the binary mask
            # Calculate the area of the first contour
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_area = cv2.contourArea(contours[0])

            # Calculate the solidity of the contour
            solidity = compute_solidity(contours[0], contour_area)

            # Check if the contour meets the area and solidity criteria
            if contour_area < nmin or contour_area > nmax or solidity < 0.8 or len(contours) != 1:
                # Set the label to 0 if it does not meet criteria (zero out the contour)
                labeling[labeling == label] = 0
            else:
                # Extract patches for the valid nuclei and store them
                patches_nuclei = get_patches(contours[0], img, label, basename, output_dir)

                # Check if any patches were obtained
                if len(patches_nuclei) == 0:
                    labeling[labeling == label] = 0  # Zero out the label if no patches are found
                else:
                    patches[label] = patches_nuclei  # Store the patches for the label
                    filtered_contours[label] = contours[0]  # Store the contour for the label
                    size_contours[label] = contour_area  # Store the contour area for the label

    # Sort the contours dictionary by label
    sorted_contours = sorted(size_contours.items(), key=lambda x: x[0])

    # Return the filtered contours, sorted contours, and patches
    return filtered_contours, sorted_contours, patches

def get_patches(contour, img, label, basename, output_dir, patch_width=512, patch_height=512):

    # Get the image height and width from the shape of the image
    image_height = img.shape[0]
    image_width = img.shape[1]

    # Calculate the bounding rectangle for the given contour
    # x, y are the coordinates of the top-left corner of the bounding box
    # w, h are the width and height of the bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate margins around the bounding box for padding, ensuring the patch stays within image boundaries
    new_margin_w_1 = (x - 0)  # Left margin (distance from left edge to the bounding box)
    new_margin_h_1 = (y - 0)  # Top margin (distance from top edge to the bounding box)

    new_margin_w_2 = (image_width - (x + w))  # Right margin (distance from right edge to the bounding box)
    new_margin_h_2 = (image_height - (y + h))  # Bottom margin (distance from bottom edge to the bounding box)

    # Use the minimum margin to avoid going outside the image boundaries
    new_margin = min(new_margin_w_1, new_margin_w_2, new_margin_h_1, new_margin_h_2)

    # Calculate the coordinates for the top-left and bottom-right corners of the patch with the new margin
    top_left_coord1 = int(x - new_margin)  # Adjust x-coordinate with the margin
    top_left_coord2 = int(y - new_margin)  # Adjust y-coordinate with the margin

    bottom_right_coord1 = int(x + w + new_margin)  # Adjust x-coordinate of the bottom-right corner
    bottom_right_coord2 = int(y + h + new_margin)  # Adjust y-coordinate of the bottom-right corner

    # Define the four corners of the patch based on the calculated coordinates
    top_left_coord = (top_left_coord1, top_left_coord2)  # Top-left corner (x, y)
    bottom_left_coord = (top_left_coord1, bottom_right_coord2)  # Bottom-left corner (x, y)
    top_right_coord = (bottom_right_coord1, top_left_coord2)  # Top-right corner (x, y)
    bottom_right_coord = (bottom_right_coord1, bottom_right_coord2)  # Bottom-right corner (x, y)

    # Extract the patch from the image using the calculated coordinates (cropping the image)
    patch = img[top_left_coord2:bottom_right_coord2, top_left_coord1:bottom_right_coord1]

    # Resize the patch to the specified dimensions (patch_width and patch_height)
    patch = cv2.resize(patch, (patch_width, patch_height), interpolation=cv2.INTER_LINEAR)

    # Generate a unique filename for the patch based on the label and basename
    patch_filename = f'{basename}_label{label}.tif'

     # Increase brightness by scaling pixel values
    patch = cv2.convertScaleAbs(patch, alpha=10.0, beta=0)

    # Save the patch to the specified output directory
    patch_output_path = os.path.join(output_dir, patch_filename)
    cv2.imwrite(patch_output_path, patch)

    # Store the patch coordinates (top-left, bottom-left, top-right, bottom-right)
    patch_dimensions = [top_left_coord, bottom_left_coord, top_right_coord, bottom_right_coord]

    # Return the list of patch coordinates so that they can be used for further processing if needed
    return patch_dimensions

def main():

    # Define and parse the command-line arguments for input parameters
    parser = argparse.ArgumentParser(description="Segmentation parameters for nuclei detection in TIFF images.")
    parser.add_argument('--nmin', '--nuclei-min-pixels', type=int, help="Minimum number of pixels for a valid nucleus.")
    parser.add_argument('--nmax', '--nuclei-max-pixels', type=int, help="Maximum number of pixels for a valid nucleus.")
    parser.add_argument('--didx', '--dapi-channel-idx', type=int, help="Index of the DAPI channel (typically 0).")
    parser.add_argument('--d', '--downsample-interval', type=int, help="Factor by which to downsample the image.")
    args = parser.parse_args()

    # Globalize the segmentation directory
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
        cancer_tif_dir = '/voronoi_tif/Cancerous'
        no_cancer_tif_dir = '/voronoi_tif/NotCancerous'
    else:
        cancer_tif_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_tif/Cancerous'
        no_cancer_tif_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_tif/NotCancerous'

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

    # Set the output directory for patches of cancerous images based on the environment (Docker or local)
    if os.path.exists('/.dockerenv'):
        cancer_patch_dir = '/voronoi/Cancerous'  # Output path for Docker environment
    else:
        cancer_patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi/Cancerous'  # Output path for local environment

    # Create the cancerous patch directory if it doesn't already exist
    os.makedirs(cancer_patch_dir, exist_ok=True)

    # Verify the creation of the cancerous patch directory and print its status
    if os.path.exists(cancer_patch_dir):
        print(f"Directory '{cancer_patch_dir}' was created successfully.")
    else:
        print(f"Failed to create the directory '{cancer_patch_dir}'.")

    # Set the output directory for patches of non-cancerous images based on the environment (Docker or local)
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
    #
    # # Perform nuclei segmentation on loaded TIFF images in the cancerous and non-cancerous datasets
    # patch_dims = segment_images(cancer_tif_paths, "Cancerous", args.nmin, args.nmax, args.didx, args.d)
    # patch_dims = segment_images(no_cancer_tif_paths, "NotCancerous", args.nmin, args.nmax, args.didx, args.d)


if __name__ == "__main__":
    main()

## Future Considerations
    # Implement parallel processing to speed up segmentation across multiple TIFF paths
    # Develop a method to precisely locate and annotate nuclei positions within the images
