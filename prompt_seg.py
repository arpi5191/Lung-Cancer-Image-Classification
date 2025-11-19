# Import required packages
import os
import torch
import shutil
import random
import tifffile
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline

# ------------------------------
# Set random seed for reproducibility
# ------------------------------
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def load_model():
    """
    Load the medical-specific Stable Diffusion model for generating synthetic
    histopathology images.

    Automatically detects GPU availability and moves the model to the appropriate
    device. If using a GPU, attention slicing is enabled to reduce memory usage
    during inference.

    Returns:
        pipe (StableDiffusionPipeline): The loaded Stable Diffusion pipeline
        device (str): The device used for generation ('cuda' or 'cpu')
    """
    # Model identifier for medical Stable Diffusion fine-tuned for histopathology
    model_id = "Nihirc/Prompt2MedImage"

    # Detect device automatically: GPU if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Medical Stable Diffusion model on {device}...")

    # Load the model with appropriate precision
    # float16 for GPU (faster), float32 for CPU (more compatible)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Disable safety checker for research purposes
    )

    # Move model to the chosen device
    pipe = pipe.to(device)

    # Enable memory-efficient attention slicing on GPU
    if device == "cuda":
        pipe.enable_attention_slicing()
        print("GPU optimizations enabled")

    print("Model loaded successfully.\n")
    return pipe, device


def train_model(pipe, device, output_dir, classifications, prompts, negative_prompts, num_images):
    """
    Generate synthetic histopathology images using the Stable Diffusion pipeline.

    Args:
        pipe (StableDiffusionPipeline): Loaded Stable Diffusion pipeline
        device (str): Device to run generation on ('cuda' or 'cpu')
        output_dir (str): Base directory to save generated images
        classifications (list of str): Labels for each class (e.g., 'Cancerous')
        prompts (list of str): Positive text prompts for image generation
        negative_prompts (list of str): Negative prompts to avoid unwanted artifacts
        num_images (list of int): Number of images to generate per class

    Returns:
        None: Saves generated images directly to disk under class-specific folders.
    """
    for class_idx in range(len(classifications)):
        # Loop through the number of images for this classification
        for img_idx in range(num_images[class_idx]):
            print(f"\n[{img_idx+1}/{num_images[class_idx]}] Generating image for prompt:")
            print(f"'{prompts[class_idx]}'")

            # Set a reproducible seed per image
            generator = torch.Generator(device=device)
            generator.manual_seed(42 + img_idx)

            # Use automatic mixed precision for faster and memory-efficient inference
            # On GPU, operations run in float16; on CPU, they run in float32
            with torch.autocast(device):

                # Generate the image using the Stable Diffusion pipeline
                result = pipe(
                    prompt=prompts[class_idx],             # Positive prompt describing what we want to generate
                    negative_prompt=negative_prompts[class_idx],  # Negative prompt specifying what to avoid (e.g., blurry, cartoon)
                    num_inference_steps=50,                 # Number of denoising steps; more steps = higher quality but slower
                    guidance_scale=10,                    # How strongly the model follows the prompt; higher = more faithful to text
                    width=512,                             # Output image width in pixels (native Stable Diffusion resolution)
                    height=512,                            # Output image height in pixels
                    generator=generator                     # Random generator for reproducibility (ensures same output for same seed)
                    )

            # Extract the generated image
            image = result.images[0]

            # Upscale to 2048x2048 using high-quality Lanczos resampling
            image = image.resize((2048, 2048), Image.Resampling.LANCZOS)

            # Convert to true grayscale
            image = image.convert("L")

            # Save the image in the class-specific folder with subfolder
            class_dir = os.path.join(output_dir, classifications[class_idx])
            image_name = f"prompt_image_{img_idx}"
            image_folder = os.path.join(class_dir, image_name)
            os.makedirs(image_folder, exist_ok=True)
            filepath = os.path.join(image_folder, f"{image_name}.tif")

            # Convert to numpy and save with tifffile
            image_array = np.array(image)
            tifffile.imwrite(filepath, image_array)
            print(f"Saved to: {filepath}")

            # Clear GPU memory after each image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nAll {sum(num_images)} images saved to {output_dir}")


def main():
    """
    Main function to set up directories, load the model, and generate images.

    Creates the output directories (removing any existing ones), defines
    prompts and classes, loads the Stable Diffusion pipeline, and calls
    the train_model function to generate images.
    """
    # Determine output directory based on environment
    if os.path.exists('/.dockerenv'):
        output_dir = '/prompt_tif'  # Use this path in Docker environment
    else:
        output_dir = '/ocean/projects/bio240001p/arpitha/prompt_tif'
        # Alternatively, local path can be used:
        # output_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/prompt_tif'

    # Remove old patch directory if it exists to start fresh
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Directory '{output_dir}' has been deleted.")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory '{output_dir}' created successfully.")

    # Create subdirectories for each class
    for label in ['Cancerous', 'NotCancerous']:
        path = os.path.join(output_dir, label)
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' was created successfully.")

    # Define number of images to generate per class
    num_images = [21, 13]

    # Define class labels
    classifications = ["Cancerous", "NotCancerous"]

    # Positive prompts: what the model should generate
    # Each string describes the desired characteristics of the synthetic histopathology image
    prompts = [
    "produce a lung histopathology grayscale image showing adenocarcinoma tumor tissue with malignant nuclei from a 66-year-old male donor.",
    "produce a lung histopathology grayscale image showing adenocarcinoma tumor tissue with benign nuclei from a 66-year-old male donor."
    ]

    # Negative prompts: features to avoid in the generated image
    # These discourage unwanted artifacts, unrealistic textures, and common mistakes in synthetic images
    negative_prompts = [
    "blurry, out of focus, low resolution, artifacts, cartoon, illustration, 3D render, text, annotations, watermark, distorted, unrealistic texture, poor contrast, overexposed, underexposed, tissue folds, air bubbles, staining artifacts",
    "blurry, out of focus, low resolution, artifacts, cartoon, illustration, 3D render, text, annotations, watermark, distorted, unrealistic texture, poor contrast, overexposed, underexposed, tissue folds, air bubbles, staining artifacts"
    ]

    # Load Stable Diffusion model
    pipe, device = load_model()

    # Generate synthetic images
    train_model(pipe, device, output_dir, classifications, prompts, negative_prompts, num_images)

if __name__ == "__main__":
    main()
