# Import necessary libraries
import os
import cv2
import shutil
import pathlib
import tifffile
import argparse
import numpy as np
from scipy.ndimage import zoom
from csbdeep.utils import normalize
import skimage.exposure as exposure
from stardist.models import StarDist2D  # Not used here but typically required if using StarDist models

def load_channel(tif_path):
    """
    Loads the specified channel (assumed to be grayscale) from a multi-channel TIFF file.

    Args:
        tif_path (str): The file path to the TIFF image.

    Returns:
        np.ndarray: The loaded and intensity-rescaled grayscale image as a 2D NumPy array.
    """
    # Read the image (assumed to be grayscale or single-channel)
    channel = tifffile.imread(tif_path)

    # Rescale intensity values to 0–255 (8-bit), which improves contrast and uniformity
    return exposure.rescale_intensity(channel, in_range='image', out_range=(0, 255)).astype(np.uint8)

def segment_images(tif_paths, classification, downsample_interval):
    """
    Segments nuclei from TIFF images using a pre-trained StarDist model and extracts patches.

    Args:
        tif_paths (List[Path]): List of paths to input .tif images.
        classification (str): Class label (e.g., 'Cancerous' or 'NotCancerous') used in output directory naming.
        downsample_interval (int): Downsampling factor (e.g., 4 means keep every 4th pixel).

    Returns:
        None
    """
    num_patches = 0  # Track total patches across all images

    for tif_path in tif_paths:
        # Extract filename without extension
        basename = tif_path.name.rstrip('.tif')

        # Load image and downsample by skipping pixels
        img = load_channel(tif_path)
        img = img[::downsample_interval, ::downsample_interval]

        # Extract and save image patches
        num_patches += extract_patches(img, classification, basename)

        # Save the processed image (after downsampling) as a .npy array
        print(f'Output Path: {seg_dir}/{basename}_seg.npy')
        np.save(f'{seg_dir}/{basename}_seg.npy', img)
        print()

    # Print summary of processing
    print("Classification:", classification)
    print("Number of Patches:", num_patches)
    print()

def extract_patches(image, classification, basename, save_size=512):
    """
    Extracts horizontal image patches and saves them as .tif files.

    Args:
        image (np.ndarray): 2D array representing the grayscale image.
        classification (str): Label used for output directory ('Cancerous' or 'NotCancerous').
        basename (str): Name of the original image (used for naming patches).
        save_size (int, optional): Size of the saved patch in pixels (default: 512).

    Returns:
        int: Number of patches saved.
    """

    # Set the output path depending on whether it's running in Docker
    if os.path.exists('/.dockerenv'):
        patch_output_dir = f'/patches/{classification}/{basename}'
    else:
        patch_output_dir = f'/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tumor_patches/{classification}/{basename}'

    # Create directory for saving patches
    os.makedirs(patch_output_dir, exist_ok=True)

    # Confirm directory creation
    if os.path.exists(patch_output_dir):
        print(f"Directory '{patch_output_dir}' created successfully.")
    else:
        print(f"Failed to create the directory '{patch_output_dir}'.")

    # Patch size and stride can be customized by classification if needed
    patch_size, stride = 256, 256

    # Get image dimensions
    h, w = image.shape[:2]
    label = 1  # Initialize patch index

    # Slide horizontally across the image
    for x in range(0, w + 1 - stride, stride):
        # Extract full-height patch at current horizontal position
        patch = image[0:h, x:x + patch_size]

        # Resize patch to standard save size (usually for model input requirements)
        patch = cv2.resize(patch, (save_size, save_size), interpolation=cv2.INTER_LINEAR)

        # Save the patch as .tif file
        patch_filename = f'{basename}_label{label}.tif'
        full_patch_path = os.path.join(patch_output_dir, patch_filename)
        cv2.imwrite(full_patch_path, patch)

        label += 1

    return (label - 1)  # Return total number of patches

def main():
    """
    Main function for performing nuclei segmentation and patch extraction
    on cancerous and non-cancerous TIFF image sets.
    """
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Segmentation parameters for nuclei detection in TIFF images.")
    parser.add_argument('--d', '--downsample-interval', type=int, required=True,
                        help="Factor by which to downsample the image (e.g., 4 = every 4th pixel).")
    args = parser.parse_args()

    global seg_dir  # Will be used inside segment_images()

    # Set segmentation directory (Docker vs local)
    if os.path.exists('/.dockerenv'):
        seg_dir = '/tumor_seg'
    else:
        seg_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tumor_seg'

    # Clear previous segmentation output directory if it exists
    if os.path.exists(seg_dir):
        shutil.rmtree(seg_dir)
        print(f"Directory '{seg_dir}' has been deleted.")
    os.makedirs(seg_dir, exist_ok=True)
    print(f"Directory '{seg_dir}' created successfully.")

    # Define input paths for both classes (Docker vs local)
    if os.path.exists('/.dockerenv'):
        cancer_tif_dir = '/tif/Cancerous'
        no_cancer_tif_dir = '/tif/NotCancerous'
    else:
        cancer_tif_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif/Cancerous'
        no_cancer_tif_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif/NotCancerous'

    # Check and load TIFF paths for both classes
    for path in [cancer_tif_dir, no_cancer_tif_dir]:
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"TIFF directory '{path}' does not exist.")

    cancer_tif_paths = list(pathlib.Path(cancer_tif_dir).glob('*.tif'))
    no_cancer_tif_paths = list(pathlib.Path(no_cancer_tif_dir).glob('*.tif'))

    if not cancer_tif_paths:
        raise FileNotFoundError(f"No .tif files found in '{cancer_tif_dir}'")
    if not no_cancer_tif_paths:
        raise FileNotFoundError(f"No .tif files found in '{no_cancer_tif_dir}'")

    print(f"Cancerous TIFF Paths: {cancer_tif_paths}")
    print(f"NotCancerous TIFF Paths: {no_cancer_tif_paths}")

    # Define patch output directory
    if os.path.exists('/.dockerenv'):
        patch_dir = '/tumor_patches'
    else:
        patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tumor_patches'

    # Remove previous patch output and recreate clean directory
    if os.path.exists(patch_dir):
        shutil.rmtree(patch_dir)
        print(f"Directory '{patch_dir}' has been deleted.")
    os.makedirs(patch_dir, exist_ok=True)
    print(f"Directory '{patch_dir}' created successfully.")

    # Create subdirectories for each class label
    for label in ['Cancerous', 'NotCancerous']:
        path = os.path.join(patch_dir, label)
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' was created successfully.")

    # Run segmentation and patch extraction for both datasets
    segment_images(cancer_tif_paths, "Cancerous", args.d)
    segment_images(no_cancer_tif_paths, "NotCancerous", args.d)

if __name__ == "__main__":
    main()
