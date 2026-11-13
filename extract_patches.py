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
    Load a single-channel grayscale TIFF image and rescale
    its pixel intensities to the 8-bit range (0-255).

    Args:
        tif_path (str or Path): Path to the input TIFF image.

    Returns:
        np.ndarray: A 2D grayscale image with uint8 pixel values.

    Raises:
        ValueError: If the input image is not a 2D grayscale image.
    """

    # Read the TIFF image while preserving its original dimensions.
    channel = tifffile.imread(tif_path)

    # Ensure that the input contains a single grayscale channel.
    if channel.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, but got shape "
            f"{channel.shape} for {tif_path}"
        )

    # Rescale the image intensities to the 0-255 range.
    # This converts the image to an 8-bit representation.
    channel = exposure.rescale_intensity(
        channel,
        in_range='image',
        out_range=(0, 255)
    ).astype(np.uint8)

    return channel


# ========================================================
# Function: prepare_square_image
# ========================================================
def prepare_square_image(image, canvas_size=2048):
    """
    Resize an input image to fit within a square canvas while
    preserving its original aspect ratio.

    If the input image is not square, zero-padding is added
    around the resized image to create a square canvas.
    The original image is neither stretched nor cropped.

    Args:
        image (np.ndarray): Input 2D grayscale image.
        canvas_size (int): Width and height of the square canvas.
            Defaults to 2048.

    Returns:
        np.ndarray: Square grayscale image with dimensions
            canvas_size x canvas_size.
    """

    # Retrieve the original image dimensions.
    h, w = image.shape[:2]

    # Calculate a scaling factor that allows the entire image
    # to fit inside the square canvas without changing its proportions.
    scale = min(canvas_size / w, canvas_size / h)

    # Calculate the resized image dimensions.
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Select an appropriate interpolation method:
    # INTER_AREA is used for downsampling.
    # INTER_LINEAR is used for upsampling.
    interpolation = (
        cv2.INTER_AREA if scale < 1
        else cv2.INTER_LINEAR
    )

    # Resize the original image while preserving its aspect ratio.
    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=interpolation
    )

    # Create a square canvas filled with zeros (black padding).
    square_image = np.zeros(
        (canvas_size, canvas_size),
        dtype=np.uint8
    )

    # Calculate offsets to center the resized image on the canvas.
    x_offset = (canvas_size - new_w) // 2
    y_offset = (canvas_size - new_h) // 2

    # Place the resized image in the center of the square canvas.
    square_image[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return square_image


# ========================================================
# Function: segment_images
# ========================================================
def segment_images(tif_paths, output_patches_dir, classification):
    """
    Process a collection of TIFF images and extract four
    square quadrant patches from each image.

    Each input image is:
        1. Loaded as a grayscale image.
        2. Rescaled to fit within a square canvas.
        3. Divided into four equal-sized quadrants.
        4. Saved as four individual TIFF patches.

    Args:
        tif_paths (List[Path]): Paths to the input TIFF images.
        output_patches_dir (str): Base directory for extracted patches.
        classification (str): Image class, either 'Cancerous'
            or 'NotCancerous'.

    Returns:
        None
    """

    # Initialize a counter for the total number of patches generated.
    num_patches = 0

    # Create the class-specific output directory path.
    output_filepath = os.path.join(
        output_patches_dir,
        classification
    )

    print(f"Saving patches to: {output_filepath}")

    # Process each TIFF image independently.
    for tif_path in tif_paths:

        # Extract the original filename without its extension.
        # Remove the suffix used for brightened tumor images, if present.
        basename = tif_path.stem.replace(
            '_brightened_image', ''
        )

        print(f"Processing: {basename}")

        # Load the original TIFF image as a grayscale image.
        img = load_channel(tif_path)

        # Fit the image within a 2048 x 2048 square canvas.
        # The original aspect ratio is preserved through padding.
        img = prepare_square_image(
            img,
            canvas_size=2048
        )

        # Extract four quadrants and save them as individual patches.
        num_patches += extract_patches(
            img,
            output_filepath,
            basename
        )

    # Display the total number of patches generated for this class.
    print(f"Classification: {classification}")
    print(f"Number of patches: {num_patches}")
    print()


# ========================================================
# Function: extract_patches
# ========================================================
def extract_patches(image, output_filepath, basename, save_size=512):
    """
    Divide a square image into four equally sized quadrants,
    resize each quadrant to the specified output dimensions,
    and save the resulting patches as TIFF files.

    The four quadrants correspond to:
        1. Top-left
        2. Top-right
        3. Bottom-left
        4. Bottom-right

    Args:
        image (np.ndarray): Square grayscale image.
        output_filepath (str): Class-specific output directory.
        basename (str): Original image filename without extension.
        save_size (int): Width and height of each output patch.
            Defaults to 512.

    Returns:
        int: Number of patches saved.

    Raises:
        ValueError: If the input image is not square or has
            odd dimensions.
    """

    # Create a separate output subdirectory for the current image.
    patch_filepath = os.path.join(
        output_filepath,
        basename
    )

    os.makedirs(patch_filepath, exist_ok=True)

    # Retrieve the dimensions of the square image.
    h, w = image.shape[:2]

    # Verify that the input image has equal height and width.
    if h != w:
        raise ValueError(
            f"Expected square image, but received shape {image.shape}"
        )

    # Ensure that the dimensions can be divided evenly into four quadrants.
    if h % 2 != 0:
        raise ValueError(
            f"Image dimensions must be even, but received {image.shape}"
        )

    # Calculate the horizontal and vertical midpoints.
    mid_y = h // 2
    mid_x = w // 2

    # Divide the square image into four non-overlapping quadrants.
    # Each quadrant has identical dimensions.
    quadrants = [
        image[:mid_y, :mid_x],       # Patch 1: Top-left
        image[:mid_y, mid_x:],       # Patch 2: Top-right
        image[mid_y:, :mid_x],       # Patch 3: Bottom-left
        image[mid_y:, mid_x:]        # Patch 4: Bottom-right
    ]

    # Process and save each quadrant independently.
    for index, patch in enumerate(quadrants, start=1):

        # Resize each square quadrant to the required output dimensions.
        # INTER_AREA is appropriate for reducing image dimensions.
        patch = cv2.resize(
            patch,
            (save_size, save_size),
            interpolation=cv2.INTER_AREA
        )

        # Generate a unique filename for the current patch.
        patch_filename = f'{basename}_label{index}.tif'

        # Construct the complete output file path.
        full_patch_path = os.path.join(
            patch_filepath,
            patch_filename
        )

        # Save the extracted patch as a TIFF image.
        tifffile.imwrite(full_patch_path, patch)

    # Return the total number of patches generated for this image.
    return len(quadrants)


# ========================================================
# Main function
# ========================================================
def main():
    """
    Execute the complete image preprocessing and patch extraction
    workflow.

    The function:
        1. Parses the requested image processing type.
        2. Determines the input and output directories.
        3. Locates TIFF images in the Cancerous and NotCancerous folders.
        4. Removes previously generated patch directories.
        5. Creates fresh class-specific output directories.
        6. Extracts four quadrant patches from every input image.

    Returns:
        None
    """

    # ----------------------------------------------------
    # Step 1: Parse command-line arguments
    # ----------------------------------------------------

    # Allow the user to select which image type to process.
    parser = argparse.ArgumentParser(
        description="Extract four quadrant patches from TIFF images."
    )

    parser.add_argument(
        '--type',
        type=str,
        choices=[
            "tumor",
            "voronoi",
            "diffusion",
            "prompt",
            "context"
        ],
        required=True,
        help="Image type to generate patches for."
    )

    # Retrieve the selected image processing type.
    args = parser.parse_args()

    # ----------------------------------------------------
    # Step 2: Determine input and output directories
    # ----------------------------------------------------

    # Select Docker paths when running inside a Docker container.
    if os.path.exists('/.dockerenv'):

        input_output_map = {
            "tumor": ("/tumor_tif", "tumor_patches"),
            "voronoi": ("/voronoi_tif", "voronoi_patches"),
            "diffusion": ("/diffusion_tif", "diffusion_patches"),
            "prompt": ("/prompt_tif", "prompt_patches"),
            "context": ("/context_tif", "context_patches")
        }

        input_patches_dir, output_patches_dir = input_output_map[args.type]

    else:

        # Define the local project directory.
        base_path = (
            "/Users/arpitha/Documents/Research_Labs/"
            "Lab_Schwartz/Lung-Cancer-Image-Classification-main"
        )

        # Map each image processing type to its corresponding
        # input TIFF directory and output patch directory.
        input_output_map = {
            "tumor": (
                f"{base_path}/tumor_tif",
                f"{base_path}/tumor_patches"
            ),
            "voronoi": (
                f"{base_path}/voronoi_tif",
                f"{base_path}/voronoi_patches"
            ),
            "diffusion": (
                f"{base_path}/diffusion_tif",
                f"{base_path}/diffusion_patches"
            ),
            "prompt": (
                f"{base_path}/prompt_tif",
                f"{base_path}/prompt_patches"
            ),
            "context": (
                f"{base_path}/context_tif",
                f"{base_path}/context_patches"
            )
        }

        # Retrieve the selected input and output directories.
        input_patches_dir, output_patches_dir = input_output_map[args.type]

    # ----------------------------------------------------
    # Step 3: Define class-specific input directories
    # ----------------------------------------------------

    # Set the directories containing cancerous and non-cancerous images.
    input_cancer_patches_dir = os.path.join(
        input_patches_dir,
        "Cancerous"
    )

    input_not_cancer_patches_dir = os.path.join(
        input_patches_dir,
        "NotCancerous"
    )

    # ----------------------------------------------------
    # Step 4: Locate Cancerous TIFF images
    # ----------------------------------------------------

    cancer_tif_path_obj = pathlib.Path(input_cancer_patches_dir)

    # Verify that the Cancerous input directory exists.
    if not cancer_tif_path_obj.exists():
        raise FileNotFoundError(
            f"TIFF directory '{input_cancer_patches_dir}' does not exist."
        )

    # Use filename patterns specific to the selected image type.
    if args.type == "tumor":

        # Locate brightened tumor images.
        cancer_tif_paths = list(
            cancer_tif_path_obj.glob('**/*_brightened_image.tif')
        )

    elif args.type == "voronoi":

        # Locate Voronoi diagram images.
        cancer_tif_paths = list(
            cancer_tif_path_obj.glob('**/*_voronoi_diagram.tif')
        )

    else:

        # Locate all TIFF images for the remaining processing types.
        cancer_tif_paths = list(
            cancer_tif_path_obj.glob('**/*.tif')
        )

    # Stop execution if no Cancerous images are found.
    if not cancer_tif_paths:
        raise FileNotFoundError(
            f"No TIFF files found in '{input_cancer_patches_dir}'."
        )

    # ----------------------------------------------------
    # Step 5: Locate NotCancerous TIFF images
    # ----------------------------------------------------

    no_cancer_tif_path_obj = pathlib.Path(
        input_not_cancer_patches_dir
    )

    # Verify that the NotCancerous input directory exists.
    if not no_cancer_tif_path_obj.exists():
        raise FileNotFoundError(
            f"TIFF directory '{input_not_cancer_patches_dir}' does not exist."
        )

    # Use the same filename selection rules as the Cancerous images.
    if args.type == "tumor":

        # Locate brightened non-cancerous images.
        no_cancer_tif_paths = list(
            no_cancer_tif_path_obj.glob('**/*_brightened_image.tif')
        )

    elif args.type == "voronoi":

        # Locate non-cancerous Voronoi diagram images.
        no_cancer_tif_paths = list(
            no_cancer_tif_path_obj.glob('**/*_voronoi_diagram.tif')
        )

    else:

        # Locate all TIFF images for the remaining processing types.
        no_cancer_tif_paths = list(
            no_cancer_tif_path_obj.glob('**/*.tif')
        )

    # Stop execution if no NotCancerous images are found.
    if not no_cancer_tif_paths:
        raise FileNotFoundError(
            f"No TIFF files found in '{input_not_cancer_patches_dir}'."
        )

    # Display the number of original images found in each class.
    print(f"Number of Cancerous TIFF files: {len(cancer_tif_paths)}")
    print(f"Number of NotCancerous TIFF files: {len(no_cancer_tif_paths)}")

    # ----------------------------------------------------
    # Step 6: Prepare output directories
    # ----------------------------------------------------

    # Remove the previous patch output directory to avoid mixing
    # newly generated patches with results from an earlier run.
    if os.path.exists(output_patches_dir):
        shutil.rmtree(output_patches_dir)

    # Create a fresh output directory.
    os.makedirs(output_patches_dir, exist_ok=True)

    # Create separate directories for the two image classes.
    for label in ['Cancerous', 'NotCancerous']:

        path = os.path.join(output_patches_dir, label)

        os.makedirs(path, exist_ok=True)

        print(f"Directory '{path}' created successfully.")

    # ----------------------------------------------------
    # Step 7: Extract patches from both image classes
    # ----------------------------------------------------

    # Process all Cancerous images and save their four quadrant patches.
    segment_images(
        cancer_tif_paths,
        output_patches_dir,
        "Cancerous"
    )

    # Process all NotCancerous images and save their four quadrant patches.
    segment_images(
        no_cancer_tif_paths,
        output_patches_dir,
        "NotCancerous"
    )


# ========================================================
# Script entry point
# ========================================================
if __name__ == "__main__":
    main()
