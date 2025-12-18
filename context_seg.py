# Import required packages
import os
import torch
import shutil
import random
import tifffile
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline

# SET CACHE DIRECTORIES TO PROJECT SPACE (NOT HOME DIRECTORY)
# This prevents disk quota errors
os.environ['HF_HOME'] = '/ocean/projects/bio240001p/arpitha/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/ocean/projects/bio240001p/arpitha/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/ocean/projects/bio240001p/arpitha/hf_cache'
os.environ['TORCH_HOME'] = '/ocean/projects/bio240001p/arpitha/torch_cache'

# ------------------------------
# Set random seed for reproducibility
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SingleClassDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for loading images of a single class (e.g., Cancerous or NotCancerous).

    Args:
        folder (str): Path to the folder containing TIFF images (or list of image paths).
        transform (callable, optional): Transformations to apply to images (e.g., resizing, normalization).
    """
    def __init__(self, folder, transform=None):
        # Handle both directory path and list of file paths
        if isinstance(folder, list):
            self.files = folder
        else:
            # Collect all .tif files from subdirectories
            self.files = []
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                if os.path.isdir(subfolder_path):
                    for f in os.listdir(subfolder_path):
                        if f.endswith(".tif"):
                            self.files.append(os.path.join(subfolder_path, f))
        self.transform = transform

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a single image from disk, apply preprocessing transforms,
        and return its index, image tensor, and filename.

        This method enables traceability between generated synthetic images
        and their original conditioning templates when used with ControlNet.

        Args:
            idx (int): Index of the image to load from the dataset.

        Returns:
            tuple:
                idx (int): Index of the image in the dataset (useful for debugging or logging).
                image_tensor (torch.Tensor): Transformed image tensor ready for model input.
                filename (str): Basename of the source image file, used for provenance tracking.
        """
        # Retrieve full path to the image
        img_path = self.files[idx]

        # Load image and convert to RGB (ensures consistent channel format)
        img = Image.open(img_path).convert("RGB")

        # Apply preprocessing transformations (resize, normalize, etc.)
        if self.transform:
            img = self.transform(img)

        # Extract filename for logging and traceability
        filename = os.path.basename(img_path)

        return idx, img, filename

def load_dataset(base_dir):
    """
    Creates PyTorch DataLoaders for cancerous and non-cancerous image datasets.

    Args:
        base_dir (str): Base directory containing "Cancerous" and "NotCancerous" subfolders.

    Returns:
        tuple: (cancer_loader, no_cancer_loader) PyTorch DataLoaders.
    """
    # Transformations: resize, convert to tensor, normalize for Stable Diffusion/ControlNet
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1] for compatibility
    ])

    # Define class-specific directories
    cancer_dir = os.path.join(base_dir, "Cancerous")
    no_cancer_dir = os.path.join(base_dir, "NotCancerous")

    # Create DataLoaders
    cancer_loader = DataLoader(SingleClassDataset(cancer_dir, transform), batch_size=21, shuffle=False)
    no_cancer_loader = DataLoader(SingleClassDataset(no_cancer_dir, transform), batch_size=13, shuffle=False)

    return cancer_loader, no_cancer_loader

def load_model():
    """
    Load a medical-specific Stable Diffusion pipeline with optional ControlNet conditioning.

    Returns:
        pipe (StableDiffusionControlNetPipeline): Pipeline capable of generating
            images conditioned on both prompts and ControlNet images.
        device (str): Device used for generation ('cuda' or 'cpu').

    Notes:
        - Automatically moves the model to GPU if available.
        - Enables attention slicing for memory-efficient inference.
        - ControlNet provides guidance from a real image to improve structural consistency.
    """
    # Model identifiers
    model_id = "Nihirc/Prompt2MedImage"
    # controlnet_id = "lllyasviel/sd-controlnet-depth"
    controlnet_id = "lllyasviel/sd-controlnet-seg"

    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Medical Stable Diffusion model on {device}...")

    # Load ControlNet model for image-based guidance
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # Load pipeline with ControlNet
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Disabled for research purposes
    )

    # Move pipeline to device
    pipe = pipe.to(device)

    # Enable GPU-specific optimization
    if device == "cuda":
        pipe.enable_attention_slicing()
        print("GPU optimizations enabled")

    print("Model loaded successfully.\n")
    return pipe, device

def train_model(pipe, device, output_dir, classifications, prompts, negative_prompts,
                num_images, loaders):
    """
    Generate synthetic histopathology images using the Stable Diffusion pipeline
    with context-engineered prompts and ControlNet conditioning.

    Args:
        pipe: StableDiffusionControlNetPipeline loaded with ControlNet
        device: Device to run generation on
        output_dir: Directory to save generated images
        classifications: List of class labels
        prompts: List of positive prompts
        negative_prompts: List of negative prompts
        num_images: Number of images to generate per class
        loaders: List of DataLoaders for each class (for ControlNet conditioning)

    Notes:
        - Context embeddings are extracted from the prompt (text guidance).
        - ControlNet uses a real image from the dataset as structural guidance.
        - Each generated image is based on **one conditioning image** from the DataLoader.
    """
    for class_idx in range(len(classifications)):
        print(f"\n{'='*60}")
        print(f"Generating {num_images[class_idx]} images for class: {classifications[class_idx]}")
        print(f"{'='*60}")

        # --- Context Engineering ---
        # Extract text embedding for the prompt
        inputs = pipe.tokenizer(
            prompts[class_idx],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            context_embedding = pipe.text_encoder(**inputs)[0]

        print(f"Context embedding shape: {context_embedding.shape}")
        print(f"Prompt: '{prompts[class_idx]}'")

        # Iterator for dataset images
        data_iter = iter(loaders[class_idx])

        for img_idx in range(num_images[class_idx]):
            print(f"\n[{img_idx+1}/{num_images[class_idx]}] Generating image...")

            # -----------------------------------------
            # Retrieve a batch for ControlNet conditioning
            # -----------------------------------------
            try:
                # Attempt to get the next batch from the DataLoader
                _, batch_images, batch_filenames = next(data_iter)
            except StopIteration:
                # If the iterator is exhausted, reinitialize it
                # This allows cycling through the dataset when
                # num_images > dataset size
                data_iter = iter(loaders[class_idx])
                _, batch_images, batch_filenames = next(data_iter)

            # -----------------------------------------
            # Select a single conditioning image
            # -----------------------------------------
            # Cycle through images inside the batch so filenames change
            batch_size = batch_images.shape[0]
            j = img_idx % batch_size

            conditioning_image = batch_images[j]
            conditioning_filename = batch_filenames[j]

            # -----------------------------------------
            # Prepare conditioning image for ControlNet
            # -----------------------------------------
            # Undo normalization: convert from [-1, 1] back to [0, 1]
            conditioning_image = (conditioning_image + 1) / 2

            # Convert tensor to PIL Image (ControlNet expects PIL input)
            conditioning_image = transforms.ToPILImage()(conditioning_image)

            # Ensure image resolution matches Stable Diffusion input size
            conditioning_image = conditioning_image.resize((512, 512))

            # Set reproducible seed
            generator = torch.Generator(device=device).manual_seed(42 + img_idx)

            # --- Image Generation ---
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                result = pipe(
                    prompt=prompts[class_idx],
                    negative_prompt=negative_prompts[class_idx],
                    image=conditioning_image,  # ControlNet guidance
                    controlnet_conditioning_scale=1.3,  # Strength of ControlNet influence
                    num_inference_steps=100,  # More steps for higher quality
                    guidance_scale=7,       # Strength of adherence to prompt
                    width=512,
                    height=512,
                    generator=generator
                )

            # --- Post-processing ---
            image = result.images[0]
            image = image.resize((2048, 2048), Image.Resampling.LANCZOS)
            image = image.convert("1")  # Convert to 1-bit black and white


            # --- Save image ---
            class_dir = os.path.join(output_dir, classifications[class_idx])
            image_name = f"context_image_{img_idx}"
            image_folder = os.path.join(class_dir, image_name)
            os.makedirs(image_folder, exist_ok=True)
            filepath = os.path.join(image_folder, f"{image_name}.tif")

            # Convert PIL Image to numpy array and save with tifffile
            image_array = np.array(image)
            tifffile.imwrite(filepath, image_array)
            print(f"Saved to: {filepath}")

            # Log template → generated mapping for traceability
            print(f"Template → Generated | {conditioning_filename} → {image_name}.tif")

            # Clear memory after each image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"All {sum(num_images)} images saved to {output_dir}")
    print(f"{'='*60}")

def main():
    """
    Main function to generate synthetic histopathology images with context engineering
    and ControlNet guidance.

    Steps:
        1. Setup output directories
        2. Load dataset DataLoaders
        3. Load Stable Diffusion + ControlNet pipeline
        4. Generate images per class using prompts and conditioning images
    """

    # --- Determine output directory ---
    if os.path.exists('/.dockerenv'):
        output_dir = '/context_tif'
    else:
        output_dir = '/ocean/projects/bio240001p/arpitha/context_tif'
        # Alternatively, local path can be used:
        # output_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/context_tif'

    # --- Clean and create output directory ---
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Directory '{output_dir}' has been deleted.")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory '{output_dir}' created successfully.")

    # --- Create subdirectories for each class ---
    for label in ['Cancerous', 'NotCancerous']:
        path = os.path.join(output_dir, label)
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' was created successfully.")

    # --- Set base dataset directory ---
    if os.path.exists('/.dockerenv'):
        base_dir = "/tif/"
    else:
        base_dir = "/ocean/projects/bio240001p/arpitha/tumor_tif"
        # Alternatively, local path can be used:
        # base_dir = "/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tumor_tif/"

    # --- Load DataLoaders ---
    cancer_loader, no_cancer_loader = load_dataset(base_dir)
    print(f"Cancerous dataset size: {len(cancer_loader.dataset)}")
    print(f"NotCancerous dataset size: {len(no_cancer_loader.dataset)}")

    loaders = [cancer_loader, no_cancer_loader]

    # --- Define generation parameters ---
    num_images = [21, 13]
    classifications = ["Cancerous", "NotCancerous"]

    # -----------------------------
    # Define prompts for each class
    # -----------------------------
    base_prompt = (
        "FFPE lung adenocarcinoma grayscale microscopy from a male, "
        "Acinar architecture, "
    )

    malignant_prompt = (
        "Malignant cells in irregular glands, "
        "Most images show tangential epithelial sections,"
        "Rarely, images have branching tubular structures with open lumens"
        "Enlarged pleomorphic hyperchromatic nuclei, "
        "Prominent nucleoli, "
        "High N/C ratio and nuclear crowding"
    )

    benign_prompt = (
        "Benign cells in intact glands, "
        "Branching tubular structures with open lumens, "
        "Nuclei lining the lumen periphery, "
        "Small uniform normochromatic nuclei, "
        "Inconspicuous nucleoli,"
        "Normal N/C ratio"
    )

    # Combined prompts for each class
    prompts = [
        base_prompt + malignant_prompt,
        base_prompt + benign_prompt
    ]

    # -----------------------------
    # Define negative prompts
    # -----------------------------
    base_negative = (
        "Blurry, low resolution, artifacts, "
        "Cartoon, illustration, drawing, sketch, "
        "Annotation, letters, text, watermark, "
        "Tissue folds, debris, "
        "Wrong tissue type"
    )

    malignant_avoid = (
        "Small, uniform, round, normochromatic nuclei, "
        "Evenly spaced and organized nuclear architecture, "
        "Uniform nuclear size, "
        "Inconspicuous nucleoli, fine chromatin, smooth nuclear membranes"
    )

    benign_avoid = (
        "Enlarged irregular hyperchromatic pleomorphic nuclei, "
        "Crowded overlapping nuclei, disorganized architecture, "
        "Variable nuclear size, "
        "Prominent nucleoli, coarse chromatin, mitotic figures"
    )

    negative_prompts = [
        base_negative + malignant_avoid,
        base_negative + benign_avoid
    ]

    # -----------------------------
    # Load Stable Diffusion pipeline
    # -----------------------------
    pipe, device = load_model()

    all_prompts = prompts + negative_prompts
    expected_tokens = 77  # Target token length for prompt validation

    # -----------------------------
    # Validate token length for all prompts
    # -----------------------------
    for i, prompt in enumerate(all_prompts):

        # Encode the prompt into token IDs, truncating if too long, and return as PyTorch tensors
        text_inputs = pipe.tokenizer(
            prompt,
            truncation=True,
            return_tensors="pt"
        )

        # Get the number of tokens in the encoded prompt tensor
        num_tokens = text_inputs.input_ids.shape[1]

        # Decode each token ID back into a string for readability/debugging
        tokens_decoded = [pipe.tokenizer.decode([tid]).strip() for tid in text_inputs.input_ids[0]]

        # Print which prompt is being validated and how many tokens it has
        print(f"Prompt {i} validated with {num_tokens} tokens.")

        # Print the decoded tokens so you can inspect the actual token sequence
        print(tokens_decoded)

        # Print an empty line to separate output for different prompts
        print()

    # -----------------------------
    # Generate synthetic images
    # -----------------------------
    train_model(pipe, device, output_dir, classifications, prompts, negative_prompts, num_images, loaders)

if __name__ == "__main__":
    main()
