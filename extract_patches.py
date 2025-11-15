import os
import cv2
import shutil
import pathlib
import tifffile
import argparse
import numpy as np
import skimage.exposure as exposure

# ========================================================
# Function: load_channel
# ========================================================
def load_channel(tif_path):
    """
    Load a single channel (grayscale) from a .tif image
    and rescale intensity to 0–255 for uniform contrast.

    Args:
        tif_path (str or Path): Path to the .tif image file.

    Returns:
        np.ndarray: 2D array representing the grayscale image.
    """
    # Read image from .tif file
    channel = tifffile.imread(tif_path)

    # Rescale intensity to 8-bit (0–255) for consistent visualization
    return exposure.rescale_intensity(channel, in_range='image', out_range=(0, 255)).astype(np.uint8)


# ========================================================
# Function: segment_images
# ========================================================
def segment_images(tif_paths, output_patches_dir, classification, downsample_interval):
    """
    Segment images and extract horizontal patches from each.

    Args:
        tif_paths (List[Path]): List of paths to input .tif files.
        output_patches_dir (str): Base directory to save patches.
        classification (str): Class label ('Cancerous' or 'NotCancerous').
        downsample_interval (int): Downsampling factor for speed.

    Returns:
        None
    """
    num_patches = 0  # Counter for total patches per class
    output_filepath = os.path.join(output_patches_dir, classification)
    print(f"Saving patches to: {output_filepath}")

    for tif_path in tif_paths:
        # Extract base filename without extension safely
        basename = tif_path.stem
        print(f"Processing: {basename}")

        # Load and downsample the image
        img = load_channel(tif_path)
        img = img[::downsample_interval, ::downsample_interval]

        # Extract and save patches
        num_patches += extract_patches(img, output_filepath, basename)

    # Summary of patches created
    print(f"Classification: {classification}")
    print(f"Number of patches: {num_patches}")
    print()


# ========================================================
# Function: extract_patches
# ========================================================
def extract_patches(image, output_filepath, basename, save_size=512):
    """
    Extract horizontal patches from the image and save as .tif files.

    Args:
        image (np.ndarray): 2D grayscale image.
        output_filepath (str): Directory to save patches.
        basename (str): Original image filename without extension.
        save_size (int, optional): Size to resize patch (default 512).

    Returns:
        int: Total number of patches saved.
    """
    # Create a subdirectory for the current image
    patch_filepath = os.path.join(output_filepath, basename)
    os.makedirs(patch_filepath, exist_ok=True)

    # Patch extraction parameters
    patch_size, stride = 256, 256
    h, w = image.shape[:2]
    label = 1  # Patch index

    # Slide horizontally to extract patches
    for x in range(0, w + 1 - stride, stride):
        patch = image[0:h, x:x + patch_size]
        patch = cv2.resize(patch, (save_size, save_size), interpolation=cv2.INTER_LINEAR)

        # Save patch as .tif
        patch_filename = f'{basename}_label{label}.tif'
        full_patch_path = os.path.join(patch_filepath, patch_filename)
        cv2.imwrite(full_patch_path, patch)

        label += 1

    return label - 1  # Total patches for this image


# ========================================================
# Main function
# ========================================================
def main():
    """
    Main workflow:
    1. Parse command-line arguments for segmentation type and downsampling.
    2. Determine input/output directories (Docker vs local).
    3. Verify presence of .tif files in Cancerous/NotCancerous folders.
    4. Remove old outputs and create fresh directories.
    5. Segment images and extract horizontal patches.
    """
    parser = argparse.ArgumentParser(description="Segmentation parameters for nuclei detection in .tif images.")

    # Segmentation mode argument
    parser.add_argument(
        '--type',
        type=str,
        choices=["tumor", "voronoi", "diffusion", "prompt", "context"],
        required=True,
        help="Segmentation type to generate patches for (tumor/voronoi/diffusion/prompt/context)."
    )

    # Downsampling factor argument
    parser.add_argument(
        '--d', '--downsample-interval',
        type=int,
        required=True,
        help="Downsample factor (e.g., 4 = keep every 4th pixel)."
    )

    args = parser.parse_args()

    # Determine input/output directories
    if os.path.exists('/.dockerenv'):
        # Docker paths
        input_output_map = {
            "tumor": ("/tif", "tumor_patches"),
            "voronoi": ("/voronoi_tif", "voronoi_patches"),
            "diffusion": ("/diffusion_tif", "diffusion_patches"),
            "prompt": ("/prompt_tif", "prompt_patches"),
            "context": ("/context_tif", "context_patches")
        }
        input_patches_dir, output_patches_dir = input_output_map[args.type]
    else:
        # Local paths
        base_path = "/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist"
        input_output_map = {
            "tumor": (f"{base_path}/tif", f"{base_path}/tumor_patches"),
            "voronoi": (f"{base_path}/voronoi_tif", f"{base_path}/voronoi_patches"),
            "diffusion": (f"{base_path}/diffusion_tif", f"{base_path}/diffusion_patches"),
            "prompt": (f"{base_path}/prompt_tif", f"{base_path}/prompt_patches"),
            "context": (f"{base_path}/context_tif", f"{base_path}/context_patches")
        }
        input_patches_dir, output_patches_dir = input_output_map[args.type]

    # Define class-specific input directories
    input_cancer_patches_dir = os.path.join(input_patches_dir, "Cancerous")
    input_not_cancer_patches_dir = os.path.join(input_patches_dir, "NotCancerous")

    # Verify that .tif files exist in each folder
    cancer_tif_path_obj = pathlib.Path(input_cancer_patches_dir)
    if not cancer_tif_path_obj.exists():
        raise FileNotFoundError(f".tif directory '{input_cancer_patches_dir}' does not exist.")
    cancer_tif_paths = list(cancer_tif_path_obj.glob('*.tif'))
    if not cancer_tif_paths:
        raise FileNotFoundError(f"No .tif files found in '{input_cancer_patches_dir}'.")

    no_cancer_tif_path_obj = pathlib.Path(input_not_cancer_patches_dir)
    if not no_cancer_tif_path_obj.exists():
        raise FileNotFoundError(f".tif directory '{input_not_cancer_patches_dir}' does not exist.")
    no_cancer_tif_paths = list(no_cancer_tif_path_obj.glob('*.tif'))
    if not no_cancer_tif_paths:
        raise FileNotFoundError(f"No .tif files found in '{input_not_cancer_patches_dir}'.")

    print(f"Number of Cancerous .tif files: {len(cancer_tif_paths)}")
    print(f"Number of NotCancerous .tif files: {len(no_cancer_tif_paths)}")

    # Remove old output directory for clean run
    if os.path.exists(output_patches_dir):
        shutil.rmtree(output_patches_dir)

    # Recreate output directory
    os.makedirs(output_patches_dir, exist_ok=True)

    # Create subdirectories for class labels
    for label in ['Cancerous', 'NotCancerous']:
        path = os.path.join(output_patches_dir, label)
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' was created successfully.")

    # Process and extract patches
    segment_images(cancer_tif_paths, output_patches_dir, "Cancerous", args.d)
    segment_images(no_cancer_tif_paths, output_patches_dir, "NotCancerous", args.d)


if __name__ == "__main__":
    main()
