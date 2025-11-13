# Import required packages
import os
import torch
import shutil
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline

class SingleClassDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for loading images of a single class (e.g., Cancerous or NotCancerous).

    Args:
        folder (str): Path to the folder containing TIFF images.
        transform (callable, optional): Transformations to apply to images (e.g., resizing, normalization).

    Notes:
        - Returns both the index and the transformed image tensor.
        - Useful for ControlNet conditioning when generating synthetic images.
    """
    def __init__(self, folder, transform=None):
        # Collect all .tif files in the folder
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tif")]
        self.transform = transform  # Store transformations to apply later

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load an image and apply transformations if specified.

        Args:
            idx (int): Index of the image to load.

        Returns:
            tuple: (idx, image_tensor)
                - idx: the index of the image (useful for tracking or debugging)
                - image_tensor: the transformed image ready for model input
        """
        # Open the image and convert to RGB
        img = Image.open(self.files[idx]).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        return idx, img

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
    cancer_loader = DataLoader(SingleClassDataset(cancer_dir, transform), batch_size=21, shuffle=True)
    no_cancer_loader = DataLoader(SingleClassDataset(no_cancer_dir, transform), batch_size=13, shuffle=True)

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

def train_model(pipe, device, patch_dir, classifications, prompts, negative_prompts,
                num_images, loaders):
    """
    Generate synthetic histopathology images using the Stable Diffusion pipeline
    with context-engineered prompts and ControlNet conditioning.

    Args:
        pipe: StableDiffusionControlNetPipeline loaded with ControlNet
        device: Device to run generation on
        patch_dir: Directory to save generated images
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

            # --- Get Conditioning Image for ControlNet ---
            try:
                _, batch_images = next(data_iter)
            except StopIteration:
                # Reset iterator if dataset exhausted
                data_iter = iter(loaders[class_idx])
                _, batch_images = next(data_iter)

            # Pick first image from batch
            conditioning_image = batch_images[0]

            # Convert tensor to PIL and denormalize for ControlNet
            conditioning_image = (conditioning_image + 1) / 2  # [-1,1] -> [0,1]
            conditioning_image = transforms.ToPILImage()(conditioning_image)
            conditioning_image = conditioning_image.resize((512, 512))  # Match pipeline input

            # Set reproducible seed
            generator = torch.Generator(device=device).manual_seed(42 + img_idx)

            # --- Image Generation ---
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                result = pipe(
                    prompt=prompts[class_idx],
                    negative_prompt=negative_prompts[class_idx],
                    image=conditioning_image,  # ControlNet guidance
                    controlnet_conditioning_scale=0.8,  # Strength of ControlNet influence
                    num_inference_steps=10,  # More steps for higher quality
                    guidance_scale=7.5,       # Strength of adherence to prompt
                    width=512,
                    height=512,
                    generator=generator
                )

            # --- Post-processing ---
            image = result.images[0]
            image = image.resize((2048, 2048), Image.Resampling.LANCZOS)
            image = image.convert("L")  # Convert to grayscale

            # --- Save image ---
            class_dir = os.path.join(patch_dir, classifications[class_idx])
            filepath = os.path.join(class_dir, f"context_image_{img_idx}.tiff")
            image.save(filepath)
            print(f"Saved to: {filepath}")

    print(f"\n{'='*60}")
    print(f"All {sum(num_images)} images saved to {patch_dir}")
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
        patch_dir = '/context_tif'
    else:
        patch_dir = '/ocean/projects/bio240001p/arpitha/context_tif'
        # Alternatively, local path can be used:
        # patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/context_tif'

    # --- Clean and create output directory ---
    if os.path.exists(patch_dir):
        shutil.rmtree(patch_dir)
        print(f"Directory '{patch_dir}' has been deleted.")
    os.makedirs(patch_dir, exist_ok=True)
    print(f"Directory '{patch_dir}' created successfully.")

    # --- Create subdirectories for each class ---
    for label in ['Cancerous', 'NotCancerous']:
        path = os.path.join(patch_dir, label)
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' was created successfully.")

    # --- Set base dataset directory ---
    if os.path.exists('/.dockerenv'):
        base_dir = "/tif/"
    else:
        base_dir = "/ocean/projects/bio240001p/arpitha/tif"
        # Alternatively, local path can be used:
        # base_dir = "/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif/"

    # --- Load DataLoaders ---
    cancer_loader, no_cancer_loader = load_dataset(base_dir)
    print(f"Cancerous dataset size: {len(cancer_loader.dataset)}")
    print(f"NotCancerous dataset size: {len(no_cancer_loader.dataset)}")

    loaders = [cancer_loader, no_cancer_loader]

    # --- Define generation parameters ---
    num_images = [21, 13]
    classifications = ["Cancerous", "NotCancerous"]

    prompts = [
        "Lung histopathology grayscale image showing malignant nuclei, high detail, realistic, microscopic view",
        "Lung histopathology grayscale image showing benign nuclei, high detail, realistic, microscopic view"
    ]

    negative_prompts = [
        "blurry, low quality, distorted, cartoon, drawing, artificial, text, watermark",
        "blurry, low quality, distorted, cartoon, drawing, artificial, text, watermark"
    ]

    # --- Load Stable Diffusion + ControlNet pipeline ---
    pipe, device = load_model()

    # --- Generate synthetic images ---
    train_model(pipe, device, patch_dir, classifications, prompts, negative_prompts,
                num_images, loaders)

if __name__ == "__main__":
    main()
