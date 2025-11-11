# Import necessary libraries
import os
import cv2
import torch
import shutil
import pathlib
import tifffile
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
from scipy.ndimage import zoom
from diffusers import UNet2DModel
import skimage.exposure as exposure
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler

def get_class_from_path(path):
    """
    Determines the class label of an image based on its directory.

    Args:
        path (str): Full file path to the image.

    Returns:
        str: Name of the parent directory representing the class.
    """
    return os.path.basename(os.path.dirname(path))


def load_channel(tif_path):
    """
    Loads a single channel (grayscale) from a multi-channel TIFF image and rescales intensity.

    Args:
        tif_path (str): The file path to the TIFF image.

    Returns:
        np.ndarray: 2D array of the grayscale image with intensity values scaled to 0-255.
    """
    # Read the image (assumed to be single-channel or grayscale)
    channel = tifffile.imread(tif_path)

    # Rescale intensity for better contrast and uniformity
    return exposure.rescale_intensity(channel, in_range='image', out_range=(0, 255)).astype(np.uint8)

class SingleClassDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for loading images of a single class (e.g., Cancerous or NotCancerous).

    Args:
        folder (str): Path to the folder containing TIFF images.
        transform (callable, optional): Transformations to apply to images (e.g., resizing, normalization).
    """
    def __init__(self, folder, transform=None):
        # Collect all .tif files in the folder
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tif")]
        self.transform = transform  # Store the transformations to apply later

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
        # Open the image and convert it to RGB (even if grayscale or multi-channel TIFF)
        img = Image.open(self.files[idx]).convert("RGB")

        # Apply the transformations (resize, normalize, etc.) if provided
        if self.transform:
            img = self.transform(img)

        # Return both the index and the image tensor
        return idx, img

def load_dataset(base_dir):
    """
    Creates PyTorch DataLoaders for cancerous and non-cancerous image datasets.

    Args:
        base_dir (str): Base directory containing "Cancerous" and "NotCancerous" subfolders.

    Returns:
        tuple: (cancer_loader, no_cancer_loader) PyTorch DataLoaders.
    """
    # Define transformations: resize, convert to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1] for CNN input
    ])

    # Define class-specific directories
    cancer_dir = os.path.join(base_dir, "Cancerous")
    no_cancer_dir = os.path.join(base_dir, "NotCancerous")

    # Create DataLoaders for both classes
    cancer_loader = DataLoader(SingleClassDataset(cancer_dir, transform), batch_size=16, shuffle=True)
    no_cancer_loader = DataLoader(SingleClassDataset(no_cancer_dir, transform), batch_size=16, shuffle=True)

    return cancer_loader, no_cancer_loader

def load_model():
    """
    Initializes and loads the main components required for a latent diffusion model:
    the Variational Autoencoder (VAE), the UNet diffusion model, and the scheduler
    controlling the noise diffusion process.

    Specifically:
    - Loads a pretrained VAE (KL Autoencoder) to map input images to and from latent space.
    - Loads a pretrained UNet, which performs the denoising steps in latent space.
    - Initializes a diffusion scheduler that controls how noise is added or removed
      during the forward and reverse diffusion processes.

    The function stores these components as global variables so that they can be used
    later in training, inference, or image generation.

    Returns:
        None
    """

    # Define device configuration
    # Use GPU ("cuda") if available, otherwise default to CPU.
    # This ensures computations run on the most efficient hardware available.
    global device, vae, unet, scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # Load the Variational Autoencoder (VAE)
    # --------------------------------------------------------------------------
    # The VAE compresses images into a *latent space* (a smaller, encoded representation),
    # and can reconstruct them back into pixel space.
    # In diffusion models like Stable Diffusion, the denoising happens in this latent space
    # rather than directly on high-resolution images (which saves a lot of computation).
    #
    # "stabilityai/sd-vae-ft-mse" is a pretrained checkpoint from the Stable Diffusion model,
    # fine-tuned to minimize mean squared error (MSE) reconstruction loss.
    #
    # The `.to(device)` ensures that the model is moved to the correct computation device.
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # --------------------------------------------------------------------------
    # Load the UNet Diffusion Model (image-only, no text embeddings)
    # --------------------------------------------------------------------------
    # The UNet is the *core denoising model* in the diffusion process.
    # It takes in a noisy latent (from the VAE) at timestep t and predicts
    # the noise component that should be removed to move toward a cleaner sample.
    #
    # Since we're not using text prompts for conditioning, we must explicitly
    # set `cross_attention_dim=None` to prevent errors related to cross-attention.
    #
    # Here, we define the UNet architecture manually:
    # - sample_size: size of the latent feature map
    # - in_channels / out_channels: latent space channels (usually 4 for Stable Diffusion)
    # - layers_per_block: number of residual layers per UNet block
    # - block_out_channels: number of channels per block
    # - down_block_types / up_block_types: type of each block
    unet = UNet2DModel(
    sample_size=32,                     # Spatial size of the input latent map (height=width=32)
    in_channels=4,                      # Number of input channels (4 = latent channels from VAE)
    out_channels=4,                     # Number of output channels (same as input latent channels)
    layers_per_block=2,                 # Number of residual layers in each down/up block
    block_out_channels=(128, 256, 512, 512),  # Number of channels for each down/up block
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),  # Types of blocks for downsampling
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")          # Types of blocks for upsampling
    ).to(device)                            # Move the model to GPU or CPU


    # --------------------------------------------------------------------------
    # Initialize the Diffusion Scheduler
    # --------------------------------------------------------------------------
    # The scheduler defines the *forward diffusion process* — how noise is added to an image
    # — and the *reverse denoising process* — how noise is gradually removed to reconstruct the image.
    #
    # - beta_start: starting variance of noise (very small → almost no noise at first).
    # - beta_end: ending variance (larger → strong noise at later timesteps).
    # - beta_schedule: how β values increase over time (scaled_linear is common).
    # - num_train_timesteps: how many diffusion steps (commonly 1000).
    #
    # LMSDiscreteScheduler is used in Stable Diffusion to handle noise scaling in latent space.
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                     num_train_timesteps=1000)

def train_model(dataloader, num_epochs=1):
    """
    Train the UNet component of a latent diffusion model on a dataset of images.

    Args:
        dataloader (torch.utils.data.DataLoader): PyTorch DataLoader yielding batches of
            (index, image_tensor) tuples.
        num_epochs (int): Number of training epochs to run.

    Notes:
        - Assumes `vae`, `unet`, `scheduler`, and `device` are already initialized globally
          via `load_model()`.
        - Uses Adam optimizer with a fixed learning rate.
        - Training loop performs a single forward and backward pass per batch:
            1. Encodes images into latent space using the VAE.
            2. Samples a random timestep t for the diffusion process.
            3. Adds Gaussian noise according to the scheduler.
            4. Predicts the noise using the UNet.
            5. Computes MSE loss between predicted and actual noise.
            6. Performs backpropagation and optimizer step.
    """

    # Initialize optimizer for UNet parameters
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

    # Loop over epochs
    for epoch in range(num_epochs):
        for batch_idx, (indices, images) in enumerate(dataloader):
            # Move image batch to GPU/CPU
            images = images.to(device)

            # ------------------------------------------------------------------
            # Step 1: Encode images into latent space using the VAE
            # ------------------------------------------------------------------
            # VAE returns a distribution; we sample from it to get latent vectors
            latents = vae.encode(images).latent_dist.sample() * 0.18215  # scaling factor used in Stable Diffusion

            # ------------------------------------------------------------------
            # Step 2: Sample random timesteps for the diffusion process
            # ------------------------------------------------------------------
            # Each image in the batch gets a random timestep t
            t = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=device).long()

            # ------------------------------------------------------------------
            # Step 3: Generate Gaussian noise and add to latents
            # ------------------------------------------------------------------
            noise = torch.randn_like(latents)  # random noise tensor
            noisy_latents = scheduler.add_noise(latents, noise, t)  # forward diffusion step

            # ------------------------------------------------------------------
            # Step 4: Predict noise using the UNet
            # ------------------------------------------------------------------
            # UNet predicts the noise component in latent space at timestep t
            noise_pred = unet(noisy_latents, t)["sample"]

            # ------------------------------------------------------------------
            # Step 5: Compute loss and perform optimization step
            # ------------------------------------------------------------------
            loss = nn.MSELoss()(noise_pred, noise)  # MSE between predicted and actual noise
            optimizer.zero_grad()  # reset gradients
            loss.backward()        # backpropagate
            optimizer.step()       # update UNet parameters

            # For demonstration/debugging, process only the first batch
            break

        # Print epoch loss for monitoring training progress
        print(f"Epoch {epoch+1}/{num_epochs} completed, loss={loss.item():.4f}")

def main():
    """
    Main function for preparing directories and loading cancerous/non-cancerous datasets.

    Steps:
    1. Determines the output patch directory depending on Docker or local environment.
    2. Deletes and recreates the patch directory to ensure a clean workspace.
    3. Creates subdirectories for 'Cancerous' and 'NotCancerous'.
    4. Loads PyTorch DataLoaders for both classes.
    """
    # Set patch output directory based on environment
    if os.path.exists('/.dockerenv'):
        patch_dir = '/gen_tif'  # Use Docker path
    else:
        patch_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/gen_tif'  # Local path

    # Remove old patch directory if it exists
    if os.path.exists(patch_dir):
        shutil.rmtree(patch_dir)
        print(f"Directory '{patch_dir}' has been deleted.")
    os.makedirs(patch_dir, exist_ok=True)  # Create fresh directory
    print(f"Directory '{patch_dir}' created successfully.")

    # Create class-specific subdirectories
    for label in ['Cancerous', 'NotCancerous']:
        path = os.path.join(patch_dir, label)
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' was created successfully.")

    # Set base dataset directory depending on environment
    if os.path.exists('/.dockerenv'):
        base_dir = "/tif/"  # Docker path
    else:
        base_dir = "/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/tif/"  # Local path

    # Load DataLoaders for cancerous and non-cancerous images
    cancer_loader, no_cancer_loader = load_dataset(base_dir)

    # Print dataset sizes for verification
    print(f"Cancerous dataset size: {len(cancer_loader.dataset)}")
    print(f"NotCancerous dataset size: {len(no_cancer_loader.dataset)}")

    load_model()  # Initialize VAE, UNet, and scheduler

    train_model(cancer_loader)

if __name__ == "__main__":
    main()
