# Import packages
import os
import cv2
import shutil
import pathlib
import tifffile
import numpy as np
from PIL import Image
import skimage.measure
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial import Voronoi, voronoi_plot_2d

# Set random seed for reproducibility
random.seed(42)            # Python random module
np.random.seed(42)

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

def process_labelling(labelling, img, minimum_solidity=0.4):
    """
    Process labeled regions in an image to filter valid nuclei based on solidity
    and compute centroids of those valid nuclei.

    Args:
        labelling (np.ndarray): 2D array with integer labels representing segmented regions.
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

            # For each contour (only one expected), validate solidity
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                solidity = compute_solidity(contour, contour_area)

                # Debug print for tracking values
                print(f"Label: {label}, Area: {contour_area}, Solidity: {solidity}")

                # Accept contour only if it meets solidity criteria
                if solidity >= minimum_solidity:
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

def voronoi_segmentation(img_paths, classification, intermediates_dir, final_dir):
    """
    Processes preprocessed images to perform nuclei segmentation and generate Voronoi diagrams.

    Parameters:
    - img_paths (list of Path): List of file paths to preprocessed images.
    - classification (str): Classification label used to organize output directories.
    - intermediates_dir (str): Directory path to save intermediate visualizations.
    - final_dir (str): Directory path to save final Voronoi diagrams.

    Process:
    - Load preprocessed images from tumor_patches
    - Perform nuclei segmentation with StarDist and refine with watershed
    - Extract centroids and generate Voronoi diagrams
    - Save intermediate visualizations and final Voronoi diagrams as TIF files
    """

    # Load pre-trained StarDist nuclei segmentation model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    for img_path in img_paths:

        # Prepare output directory paths
        basename = img_path.stem.replace('_brightened_image', '')  # Remove suffix to get original basename
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

        # Load preprocessed image (already preprocessed, no need for CLAHE/blur/closing)
        preprocessed_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if preprocessed_img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Predict nuclei masks with StarDist
        labelling, _ = model.predict_instances(normalize(preprocessed_img), prob_thresh=0.5, nms_thresh=0.3)

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
            print(f"No nuclei detected in {img_path}")
            continue

        # Generate and save label visualization (intermediate)
        fig, ax = plt.subplots(figsize=(8, 8))
        labeled_img = label2rgb(labelling, bg_label=0)
        ax.imshow(labeled_img)
        ax.axis('off')
        label_vis_path = os.path.join(intermediates_output_dir, f"{basename}_labelling_diagram.png")
        plt.savefig(label_vis_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Labeled diagram saved at {label_vis_path}")

        # Extract centroids and contours from labelling
        centroids, contours = process_labelling(labelling, preprocessed_img)

        if len(centroids) < 4:
            print(f"Not enough centroids ({len(centroids)}) for Voronoi diagram in {img_path}")
            continue

        # Visualize contours and save (intermediate)
        contour_canvas = np.zeros_like(labelling, dtype=np.uint8)
        for label, contour_list in contours.items():
            cv2.drawContours(contour_canvas, contour_list, -1, (255, 255, 255), thickness=4)
        contours_path = os.path.join(intermediates_output_dir, f"{basename}_contours_visualization.png")
        cv2.imwrite(contours_path, contour_canvas)
        print(f"Contours visualization saved at {contours_path}")

        # Visualize centroids and save (intermediate)
        centroids_canvas = np.zeros_like(labelling, dtype=np.uint8)
        for centroid in np.array(centroids):
            x, y = map(int, centroid)
            if 0 <= x < centroids_canvas.shape[1] and 0 <= y < centroids_canvas.shape[0]:
                cv2.circle(centroids_canvas, (x, y), 5, (255, 255, 255), thickness=-1)
        centroids_path = os.path.join(intermediates_output_dir, f"{basename}_centroids_visualization.png")
        cv2.imwrite(centroids_path, centroids_canvas)
        print(f"Centroids visualization saved at {centroids_path}")

        # Generate Voronoi diagram from centroids
        fig, ax = plt.subplots(figsize=(8, 8))
        vor = Voronoi(np.array(centroids))
        voronoi_plot_2d(vor, ax=ax, show_points=True, show_vertices=True, line_colors="orange", line_width=2)
        ax.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='red', s=30, marker="o")
        img_height, img_width = labelling.shape
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # flip y-axis for image coordinate system
        ax.axis('off')

        # Save Voronoi diagram as PNG in intermediates
        voronoi_intermediates_png_path = os.path.join(intermediates_output_dir, f"{basename}_voronoi_diagram.png")
        fig.savefig(voronoi_intermediates_png_path, bbox_inches='tight', dpi=500, format='png')
        print(f"Voronoi diagram (PNG) saved at {voronoi_intermediates_png_path}")

        plt.close(fig)

        # --- Save Voronoi diagram as TIF WITHOUT ANY WHITESPACE ---

        # Convert the previously saved PNG Voronoi diagram to a TIF file for downstream processing or compatibility
        voronoi_final_tif_path = os.path.join(final_output_dir, f"{basename}_voronoi_diagram.tif")
        png_img = Image.open(voronoi_intermediates_png_path)
        png_img.save(voronoi_final_tif_path, format="TIFF")
        print(f"Voronoi diagram (TIF) saved at {voronoi_final_tif_path}")

        # Load images for overlaying
        original_image = Image.open(str(img_path)).convert("RGBA")
        label_vis_img = Image.open(label_vis_path).convert("RGBA")
        contours_img = Image.open(contours_path).convert("RGBA")
        centroids_img = Image.open(centroids_path).convert("RGBA")
        voronoi_img = Image.open(voronoi_intermediates_png_path).convert("RGBA")

        # Resize label visualization to match original image if needed
        if original_image.size != label_vis_img.size:
            print(f"Resizing label visualization from {label_vis_img.size} to {original_image.size}")
            label_vis_img = label_vis_img.resize(original_image.size)

        # Combine original image and label visualization side-by-side and save
        original_labelled_img = np.hstack((np.array(original_image), np.array(label_vis_img)))
        orig_labelled_path = os.path.join(intermediates_output_dir, f"{basename}_original_labelled_image.png")
        cv2.imwrite(orig_labelled_path, original_labelled_img)

        # Resize contours visualization to match label visualization if needed
        if label_vis_img.size != contours_img.size:
            print(f"Resizing contours visualization from {contours_img.size} to {label_vis_img.size}")
            contours_img = contours_img.resize(label_vis_img.size)

        # Combine label visualization and contours visualization side-by-side and save
        labelled_contours_img = np.hstack((np.array(label_vis_img), np.array(contours_img)))
        labelled_contours_path = os.path.join(intermediates_output_dir, f"{basename}_labelled_contours_image.png")
        cv2.imwrite(labelled_contours_path, labelled_contours_img)

        # Resize centroids visualization to match original image if needed
        if original_image.size != centroids_img.size:
            print(f"Resizing centroids visualization from {centroids_img.size} to {original_image.size}")
            centroids_img = centroids_img.resize(original_image.size)

        # Change centroid pixels to red color
        centroids_img.putdata([(0, 0, 255, 255) if pixel[0] != 0 else pixel for pixel in centroids_img.getdata()])

        # Blend original image and centroids visualization and save
        original_centroids_img = Image.blend(original_image, centroids_img, alpha=0.4)
        original_centroids_path = os.path.join(intermediates_output_dir, f"{basename}_original_centroids_image.png")
        original_centroids_img.save(original_centroids_path, format="PNG")

        # Resize centroids visualization to match contours visualization if needed
        if contours_img.size != centroids_img.size:
            print(f"Resizing centroids visualization from {centroids_img.size} to {contours_img.size}")
            centroids_img = centroids_img.resize(contours_img.size)

        # Blend contours and centroids visualizations and save
        contours_centroids_img = Image.blend(contours_img, centroids_img, alpha=0.4)
        contours_centroids_path = os.path.join(intermediates_output_dir, f"{basename}_contours_centroids_image.png")
        contours_centroids_img.save(contours_centroids_path, format="PNG")

        # Resize Voronoi diagram to match centroids visualization if needed
        if centroids_img.size != voronoi_img.size:
            print(f"Resizing Voronoi diagram from {voronoi_img.size} to {centroids_img.size}")
            voronoi_img = voronoi_img.resize(centroids_img.size)

        # Blend centroids visualization and Voronoi diagram and save
        voronoi_centroids_img = Image.blend(centroids_img, voronoi_img, alpha=0.2)
        voronoi_centroids_path = os.path.join(intermediates_output_dir, f"{basename}_voronoi_centroids_image.png")
        voronoi_centroids_img.save(voronoi_centroids_path, format="PNG")

        # Resize Voronoi diagram to match original image if needed
        if original_image.size != voronoi_img.size:
            print(f"Resizing Voronoi diagram from {voronoi_img.size} to {original_image.size}")
            voronoi_img = voronoi_img.resize(original_image.size)

        # Save Voronoi diagram only (no blend with original)
        original_voronoi_img = voronoi_img
        original_voronoi_path = os.path.join(intermediates_output_dir, f"{basename}_original_voronoi_image.png")
        original_voronoi_img.save(original_voronoi_path, format="PNG")
        print(f"Labeled diagram saved at {label_vis_path}")

def main():
    """
    Main entry point for Voronoi segmentation pipeline.

    This function:
    - Sets up input directories for preprocessed Cancerous and NotCancerous images from tumor_patches.
    - Validates the presence of image files in these directories.
    - Sets up and cleans output directories for Voronoi results.
    - Calls the Voronoi segmentation function on cancerous and non-cancerous image lists.
    """

    # Determine input directories for preprocessed images based on environment
    if os.path.exists('/.dockerenv'):
        cancer_img_dir = '/tumor_tif/Cancerous'
        no_cancer_img_dir = '/tumor_tif/NotCancerous'
    else:
        cancer_img_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tumor_tif/Cancerous'
        no_cancer_img_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tumor_tif/NotCancerous'

    # Verify and list Cancerous preprocessed images
    cancer_img_path_obj = pathlib.Path(cancer_img_dir)
    if not cancer_img_path_obj.exists():
        raise FileNotFoundError(f"Image directory '{cancer_img_dir}' does not exist.")
    cancer_img_paths = list(cancer_img_path_obj.glob('*/*_brightened_image.tif'))
    if not cancer_img_paths:
        raise FileNotFoundError(f"No preprocessed images found in '{cancer_img_dir}'.")
    print(f"Cancerous Image Paths: {cancer_img_paths}")

    # Verify and list NotCancerous preprocessed images
    no_cancer_img_path_obj = pathlib.Path(no_cancer_img_dir)
    if not no_cancer_img_path_obj.exists():
        raise FileNotFoundError(f"Image directory '{no_cancer_img_dir}' does not exist.")
    no_cancer_img_paths = list(no_cancer_img_path_obj.glob('*/*_brightened_image.tif'))
    if not no_cancer_img_paths:
        raise FileNotFoundError(f"No preprocessed images found in '{no_cancer_img_dir}'.")
    print(f"NotCancerous Image Paths: {no_cancer_img_paths}")

    # Setup voronoi_intermediates output directory and clean if exists
    voronoi_intermediates_dir = '/voronoi_intermediates' if os.path.exists('/.dockerenv') else \
        '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_intermediates'
    if os.path.exists(voronoi_intermediates_dir):
        shutil.rmtree(voronoi_intermediates_dir)
        print(f"Directory '{voronoi_intermediates_dir}' has been deleted.")
    os.makedirs(voronoi_intermediates_dir, exist_ok=True)
    print(f"Directory '{voronoi_intermediates_dir}' was created successfully." if os.path.exists(voronoi_intermediates_dir) else
          f"Failed to create the directory '{voronoi_intermediates_dir}'.")

    # Setup voronoi_tif output directory and clean if exists
    voronoi_tif_dir = '/voronoi_tif' if os.path.exists('/.dockerenv') else \
        '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_tif'
    if os.path.exists(voronoi_tif_dir):
        shutil.rmtree(voronoi_tif_dir)
        print(f"Directory '{voronoi_tif_dir}' has been deleted.")
    os.makedirs(voronoi_tif_dir, exist_ok=True)
    print(f"Directory '{voronoi_tif_dir}' was created successfully." if os.path.exists(voronoi_tif_dir) else
          f"Failed to create the directory '{voronoi_tif_dir}'.")

    # Print the number of images in each category
    print(f"Number of Cancerous Images: {len(cancer_img_paths)}")
    print(f"Number of NotCancerous Images: {len(no_cancer_img_paths)}")

    # Run the Voronoi segmentation pipeline for cancerous and non-cancerous images
    voronoi_segmentation(cancer_img_paths, "Cancerous", voronoi_intermediates_dir, voronoi_tif_dir)
    voronoi_segmentation(no_cancer_img_paths, "NotCancerous", voronoi_intermediates_dir, voronoi_tif_dir)

if __name__ == "__main__":
    main()
