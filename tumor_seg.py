# Import packages
import os
import cv2
import shutil
import pathlib
import tifffile
import argparse
import numpy as np
import skimage.exposure as exposure

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

def preprocess_images(tif_paths, classification, intermediates_dir, final_dir, dapi_channel_idx, downsample_interval):
    """
    Preprocesses a list of TIFF images and saves the results.

    Parameters:
    - tif_paths (list of Path): List of file paths to multi-channel TIFF images.
    - classification (str): Classification label used to organize output directories.
    - intermediates_dir (str): Directory path to save intermediate preprocessing steps.
    - final_dir (str): Directory path to save final preprocessed results.
    - dapi_channel_idx (int): Index of the DAPI channel in the TIFF images.
    - downsample_interval (int): Factor by which to downsample the input images for processing.

    Process:
    - Load DAPI channel images
    - Downsample images
    - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Apply Gaussian blur
    - Apply morphological closing
    - Save intermediate and final preprocessed images
    """

    for tif_path in tif_paths:

        # Prepare output directory paths
        basename = tif_path.stem  # Get the filename without extension to use as subfolder name
        intermediates_output_dir = os.path.join(intermediates_dir, classification, basename)
        final_output_dir = os.path.join(final_dir, classification, basename)

        # Create the directories if they don't already exist
        os.makedirs(intermediates_output_dir, exist_ok=True)
        os.makedirs(final_output_dir, exist_ok=True)

        # Check and confirm directory creation
        if os.path.exists(intermediates_output_dir):
            print(f"Directory '{intermediates_output_dir}' created successfully.")
        else:
            print(f"Failed to create the directory '{intermediates_output_dir}'.")

        if os.path.exists(final_output_dir):
            print(f"Directory '{final_output_dir}' created successfully.")
        else:
            print(f"Failed to create the directory '{final_output_dir}'.")

        # Load and downsample DAPI channel image
        original_img = load_channel(tif_path, dapi_channel_idx)
        original_img = original_img[::downsample_interval, ::downsample_interval]

        # Save original downsampled image (intermediate only)
        original_img_path = os.path.join(intermediates_output_dir, f"{basename}_original_image.png")
        cv2.imwrite(original_img_path, original_img)
        print(f"Original image saved at {original_img_path}")

        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if original_img.dtype != np.uint8:
            original_img = ((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255).astype(np.uint8)
        preprocessed_img = clahe.apply(original_img)

        # Smooth image with Gaussian Blur to reduce noise
        preprocessed_img = cv2.GaussianBlur(preprocessed_img, (3, 3), 0)

        # Apply morphological closing to reduce noise
        kernel = np.ones((7, 7), np.uint8)
        preprocessed_img = cv2.morphologyEx(preprocessed_img, cv2.MORPH_CLOSE, kernel)

        # Save preprocessed image (intermediate only)
        preprocessed_img_path = os.path.join(intermediates_output_dir, f"{basename}_preprocessed_image.png")
        cv2.imwrite(preprocessed_img_path, preprocessed_img)
        print(f"Preprocessed image saved at {preprocessed_img_path}")

        # Create brightened version for visualization
        brightness_increase = 30
        bright_img = cv2.convertScaleAbs(preprocessed_img, alpha=2, beta=brightness_increase)

        # Save brightened image in both intermediates and final directories
        bright_img_intermediates_path = os.path.join(intermediates_output_dir, f"{basename}_brightened_image.png")
        cv2.imwrite(bright_img_intermediates_path, bright_img)
        print(f"Brightened image saved at {bright_img_intermediates_path}")

        bright_img_final_path = os.path.join(final_output_dir, f"{basename}_brightened_image.tif")
        tifffile.imwrite(bright_img_final_path, bright_img)
        print(f"Final brightened image saved at {bright_img_final_path}")

def main():
    """
    Main entry point for image preprocessing pipeline.

    This function:
    - Parses command-line arguments for DAPI channel index and downsampling factor.
    - Sets up input TIFF directories for Cancerous and NotCancerous samples based on the environment (Docker/local).
    - Validates the presence of TIFF files in these directories.
    - Sets up and cleans output directories for intermediate and final preprocessing results.
    - Calls the preprocessing function on cancerous and non-cancerous TIFF image lists.
    """

    # Parse command-line arguments for preprocessing parameters
    parser = argparse.ArgumentParser(description="Preprocessing parameters for TIFF images.")
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
        cancer_tif_dir = '/ocean/projects/bio240001p/arpitha/tif/Cancerous'
        no_cancer_tif_dir = '/ocean/projects/bio240001p/arpitha/tif/NotCancerous'
        # Alternatively
        # cancer_tif_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif/Cancerous'
        # no_cancer_tif_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif/NotCancerous'

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

    # Setup tif_intermediates output directory and clean if exists
    tif_intermediates_dir = '/tif_intermediates' if os.path.exists('/.dockerenv') else \
        '/ocean/projects/bio240001p/arpitha/tif_intermediates'
    # Alternatively
    # tif_intermediates_dir = '/tif_intermediates' if os.path.exists('/.dockerenv') else \
    #     '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif_intermediates'
    if os.path.exists(tif_intermediates_dir):
        shutil.rmtree(tif_intermediates_dir)
        print(f"Directory '{tif_intermediates_dir}' has been deleted.")
    os.makedirs(tif_intermediates_dir, exist_ok=True)
    print(f"Directory '{tif_intermediates_dir}' was created successfully." if os.path.exists(tif_intermediates_dir) else
          f"Failed to create the directory '{tif_intermediates_dir}'.")

    # Setup tumor_patches output directory and clean if exists
    tumor_output_dir = '/tumor_tif' if os.path.exists('/.dockerenv') else \
        '/ocean/projects/bio240001p/arpitha/tumor_tif'
    # # Alternatively
    # tumor_output_dir = '/tumor_tif' if os.path.exists('/.dockerenv') else \
    #     '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tumor_tif'
    if os.path.exists(tumor_output_dir):
        shutil.rmtree(tumor_output_dir)
        print(f"Directory '{tumor_output_dir}' has been deleted.")
    os.makedirs(tumor_output_dir, exist_ok=True)
    print(f"Directory '{tumor_output_dir}' was created successfully." if os.path.exists(tumor_output_dir) else
          f"Failed to create the directory '{tumor_output_dir}'.")

    # Print the number of TIFF images in each category
    print(f"Number of Cancerous TIFFs: {len(cancer_tif_paths)}")
    print(f"Number of NotCancerous TIFFs: {len(no_cancer_tif_paths)}")

    # Run the preprocessing pipeline for cancerous and non-cancerous images
    preprocess_images(cancer_tif_paths, "Cancerous", tif_intermediates_dir, tumor_output_dir, args.didx, args.d)
    preprocess_images(no_cancer_tif_paths, "NotCancerous", tif_intermediates_dir, tumor_output_dir, args.didx, args.d)

if __name__ == "__main__":
    main()
