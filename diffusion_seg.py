# ------------------------------
# IMPORTS
# ------------------------------
import os
import gc  # for manual garbage collection
import torch
import shutil  # for file operations
import random
import tifffile  # for reading/writing .tif images
import numpy as np
from PIL import Image
import torch.nn as nn
from scipy.ndimage import zoom  # for upscaling images
from diffusers import UNet2DModel, DDIMScheduler, AutoencoderKL
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Optimize PyTorch GPU memory allocation (allow segments to expand)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ------------------------------
# RANDOM SEEDS FOR REPRODUCIBILITY
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------
# CUSTOM DATASET
# ------------------------------
class SingleClassDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading images of a single class.

    Supports:
    - Loading all .tif images recursively from subfolders in a directory.
    - Using a pre-defined list of file paths directly.
    """
    def __init__(self, folder, transform=None):
        # If folder is already a list of files, use it directly
        if isinstance(folder, list):
            self.files = folder
        else:
            self.files = []
            # Iterate through subfolders and collect all .tif files
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                if os.path.isdir(subfolder_path):
                    for f in os.listdir(subfolder_path):
                        if f.endswith(".tif"):
                            self.files.append(os.path.join(subfolder_path, f))
        self.transform = transform  # optional image transformations

    def __len__(self):
        # Return total number of images
        return len(self.files)

    def __getitem__(self, idx):
        # Load image at index and convert to RGB
        img = Image.open(self.files[idx]).convert("RGB")
        # Apply optional transforms (e.g., resize, normalize)
        if self.transform:
            img = self.transform(img)
        # Return index and processed image
        return idx, img

# ------------------------------
# DATA LOADER FUNCTION
# ------------------------------
def load_dataset(base_dir):
    """
    Creates PyTorch DataLoaders for cancerous and non-cancerous images.

    Images are:
    - Resized to 256x256
    - Converted to tensors
    - Normalized to [-1, 1] for training

    Small batch sizes (4) are used to reduce GPU memory usage.
    """

    transform = transforms.Compose([
        transforms.Resize((256, 256)),               # resize all images to 256x256
        transforms.RandomHorizontalFlip(p=0.5),      # Random flips
        transforms.RandomVerticalFlip(p=0.5),        # Random flips
        transforms.RandomRotation(degrees=15),       # Random rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Slight color variation
        transforms.ToTensor(),                       # convert PIL image to PyTorch tensor
        transforms.Normalize([0.5], [0.5])           # scale pixel values to [-1, 1]
    ])

    # Define dataset directories
    cancer_dir = os.path.join(base_dir, "Cancerous")
    no_cancer_dir = os.path.join(base_dir, "NotCancerous")

    # Create DataLoaders with shuffle=True for random batching
    cancer_loader = DataLoader(SingleClassDataset(cancer_dir, transform),
                               batch_size=4, shuffle=True)
    no_cancer_loader = DataLoader(SingleClassDataset(no_cancer_dir, transform),
                                  batch_size=4, shuffle=True)

    return cancer_loader, no_cancer_loader

# ------------------------------
# MODEL LOADING FUNCTION
# ------------------------------
def load_model():
    """
    Loads:
        1. AutoencoderKL (VAE) for encoding/decoding images into latent space
        2. UNet2DModel with class conditioning for diffusion
        3. DDIM scheduler for iterative denoising

    Notes:
        - VAE is frozen and kept in eval mode for memory efficiency
        - UNet is conditioned on class labels (cancerous/non-cancerous)
        - Scheduler is used during training; fresh schedulers are created during generation
    """
    global device, vae, unet, scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clear GPU memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Load pre-trained VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False  # freeze VAE

    # Initialize class-conditioned UNet
    unet = UNet2DModel(
        sample_size=32,  # spatial size of latent
        in_channels=4,  # latent channels
        out_channels=4,  # latent channels output
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        class_embed_type="timestep",  # class embedding style
        num_class_embeds=2  # 0=cancerous, 1=non-cancerous
    ).to(device)

    # Scheduler for training (used for adding noise during forward diffusion)
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000
    )

    print(f"Models loaded on {device}")
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")

# ------------------------------
# TRAINING FUNCTION
# ------------------------------
def train_model(dataloader, num_epochs=3000, class_label=0):
    """
    Trains the UNet with class-conditioned latent diffusion.

    Args:
        dataloader: PyTorch DataLoader providing batches of images for one class
        num_epochs: total training epochs
        class_label: 0=cancerous, 1=non-cancerous
    """
    # ------------------------------
    # Optimizer and scheduler setup
    # ------------------------------
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)  # Adam optimizer for UNet
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=num_epochs)  # Cosine annealing LR schedule
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training to save GPU memory

    print(f"\nTraining class {class_label} ({'Cancerous' if class_label == 0 else 'Non-Cancerous'})")

    # ------------------------------
    # Epoch loop
    # ------------------------------
    for epoch in range(num_epochs):
        epoch_loss = 0  # Sum of losses over batches
        num_batches = 0  # Counter for averaging loss

        # ------------------------------
        # Batch loop
        # ------------------------------
        for batch_idx, (indices, images) in enumerate(dataloader):
            images = images.to(device)  # Move batch to GPU

            # ------------------------------
            # Encode images to latent space using frozen VAE
            # ------------------------------
            with torch.no_grad():  # VAE not trained, no gradients needed
                latents = vae.encode(images).latent_dist.sample() * 0.18215
                # Multiply by 0.18215 to scale latent appropriately for Stable Diffusion

            # ------------------------------
            # Sample random timesteps for diffusion
            # ------------------------------
            t = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()
            # Each image in the batch gets a random timestep to simulate noise addition at various diffusion stages

            # ------------------------------
            # Add Gaussian noise to latents
            # ------------------------------
            noise = torch.randn_like(latents)  # Random noise tensor
            noisy_latents = scheduler.add_noise(latents, noise, t)
            # noisy_latents = latents + noise according to the diffusion timestep t

            # ------------------------------
            # Prepare class labels for conditional generation
            # ------------------------------
            class_labels = torch.full(
                (latents.shape[0],),
                class_label,
                dtype=torch.long,
                device=device
            )

            # ------------------------------
            # Forward pass with mixed precision
            # ------------------------------
            with torch.cuda.amp.autocast():
                # UNet predicts the noise added to the latents
                noise_pred = unet(noisy_latents, t, class_labels=class_labels).sample
                # Compute loss: difference between predicted noise and actual noise
                loss = nn.MSELoss()(noise_pred, noise)

            # ------------------------------
            # Backpropagation
            # ------------------------------
            optimizer.zero_grad()            # Clear previous gradients
            scaler.scale(loss).backward()    # Scale loss and compute gradients
            scaler.unscale_(optimizer)      # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)  # Prevent exploding gradients
            scaler.step(optimizer)           # Update weights
            scaler.update()                  # Update the scaler for next iteration

            # ------------------------------
            # Track epoch loss
            # ------------------------------
            epoch_loss += loss.item()
            num_batches += 1

            # ------------------------------
            # Free GPU memory periodically
            # ------------------------------
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

        # ------------------------------
        # Update learning rate
        # ------------------------------
        scheduler_lr.step()
        avg_loss = epoch_loss / num_batches

        # ------------------------------
        # Print progress every 50 epochs
        # ------------------------------
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler_lr.get_last_lr()[0]:.6f}")

# ------------------------------
# IMAGE GENERATION FUNCTION
# ------------------------------
def generate_synthetic(num_images: int, classification, output_dir, class_label=0):
    """
    Generates synthetic images using the trained UNet and frozen VAE.

    CRITICAL FIX: Creates a FRESH scheduler for each image to prevent state pollution
    that can cause duplicate or similar images. Each image uses:
    - A unique random seed (SEED + image_index + class_offset)
    - An independent scheduler instance (no shared state between images)

    Process:
        - Start from unique random latent noise
        - Iteratively denoise using DDIM scheduler with class conditioning
        - Decode latents to pixel space using VAE
        - Convert RGB output to grayscale
        - Upscale to target resolution and save as TIFF

    Args:
        num_images: Number of images to generate
        classification: Folder name ("Cancerous" or "NotCancerous")
        output_dir: Base output directory
        class_label: 0=cancerous, 1=non-cancerous
    """
    img_size = 256
    target_size = 2048

    # Set models to evaluation mode (disable dropout, batchnorm updates)
    vae.eval()
    unet.eval()

    print(f"\nGenerating {num_images} {classification} images...")

    # ------------------------------
    # Loop for each image to generate
    # ------------------------------
    with torch.no_grad():  # No gradient computation needed
        for i in range(num_images):
            print(f"Generating image {i+1}/{num_images}...")

            # ------------------------------
            # CRITICAL FIX: Create a FRESH scheduler for THIS image only
            # This prevents scheduler state pollution between images
            # ------------------------------
            gen_scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            )
            # Set timesteps for this scheduler instance
            gen_scheduler.set_timesteps(50)  # 50 steps for quality generation

            # ------------------------------
            # Create unique random seed for this image
            # SEED + i ensures each image is different
            # class_label * 10000 ensures cancerous and non-cancerous don't overlap
            # ------------------------------
            generator = torch.Generator(device=device)
            # generator.manual_seed(torch.randint(0, 1_000_000, (1,)).item())
            generator.manual_seed(SEED + i + class_label * 10000)

            # ------------------------------
            # Initialize unique latent noise for this image using the seeded generator
            # ------------------------------
            latents = torch.randn(
                (1, 4, img_size//8, img_size//8),
                generator=generator,
                device=device
            )

            # Prepare class label for conditioning
            class_labels = torch.full((1,), class_label, dtype=torch.long, device=device)

            # ------------------------------
            # Iterative denoising loop using the FRESH scheduler
            # ------------------------------
            for t in gen_scheduler.timesteps:
                # Scale latent according to current timestep
                latent_model_input = gen_scheduler.scale_model_input(latents, t)

                # UNet predicts noise at this timestep with class conditioning
                noise_pred = unet(latent_model_input, t, class_labels=class_labels).sample

                # Scheduler removes predicted noise from latent
                # Using gen_scheduler ensures no interference between images
                latents = gen_scheduler.step(noise_pred, t, latents).prev_sample

            # ------------------------------
            # Decode latent to image pixels using VAE
            # ------------------------------
            image = vae.decode(latents / 0.18215).sample  # Decode and rescale
            image = (image.clamp(-1, 1) + 1) / 2  # Normalize to [0,1]

            # ------------------------------
            # Convert to grayscale
            # ------------------------------
            image_np = image.squeeze(0).cpu().numpy()  # Remove batch dim
            grayscale = image_np.mean(axis=0)  # Average RGB channels

            # ------------------------------
            # Upscale to target size
            # ------------------------------
            if grayscale.shape != (target_size, target_size):
                zoom_factor = target_size / grayscale.shape[0]
                grayscale = zoom(grayscale, zoom_factor, order=3)  # Cubic spline interpolation

            # ------------------------------
            # Save image as uint8 TIFF
            # ------------------------------
            grayscale = (grayscale * 255).astype(np.uint8)
            class_dir = os.path.join(output_dir, classification)
            image_name = f"generated_image_{i}"
            image_folder = os.path.join(class_dir, image_name)
            os.makedirs(image_folder, exist_ok=True)
            filepath = os.path.join(image_folder, f"{image_name}.tif")
            tifffile.imwrite(filepath, grayscale)
            print(f"Saved: {filepath}")

            # ------------------------------
            # Free GPU memory to avoid OOM
            # ------------------------------
            torch.cuda.empty_cache()

# ------------------------------
# MAIN PIPELINE
# ------------------------------
def main():
    """
    Main pipeline:
        1. Prepare output directories
        2. Load datasets
        3. Train UNet for cancerous/non-cancerous images with class conditioning
        4. Generate synthetic images with unique seeds and fresh schedulers
    """
    # Output directories depending on environment
    if os.path.exists('/.dockerenv'):
        output_dir = '/diffusion_tif'
    else:
        output_dir = '/ocean/projects/bio240001p/arpitha/diffusion_tif'

    # Remove old output dir if exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Directory '{output_dir}' deleted.")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory '{output_dir}' created.")

    # Create subfolders for each class
    for label in ['Cancerous', 'NotCancerous']:
        path = os.path.join(output_dir, label)
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created.")

    # Base dataset folder
    if os.path.exists('/.dockerenv'):
        base_dir = "/tumor_tif/"
    else:
        base_dir = '/ocean/projects/bio240001p/arpitha/tumor_tif'

    # Load datasets
    cancer_loader, no_cancer_loader = load_dataset(base_dir)
    print(f"\nCancerous dataset size: {len(cancer_loader.dataset)}")
    print(f"NotCancerous dataset size: {len(no_cancer_loader.dataset)}")

    # -------------------------------------------------------------------------
    # Phase 1: Cancerous images (class_label=0)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 1: CANCEROUS IMAGES")
    print("="*60)
    load_model()
    train_model(cancer_loader, num_epochs=1000, class_label=0)
    generate_synthetic(21, "Cancerous", output_dir, class_label=0)

    # Optional: Save trained model
    # torch.save(unet.state_dict(), "unet_cancerous_final.pth")
    # print("Cancerous model saved: unet_cancerous_final.pth")

    # -------------------------------------------------------------------------
    # Phase 2: Non-cancerous images (class_label=1)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 2: NON-CANCEROUS IMAGES")
    print("="*60)
    load_model()  # Reinitialize with fresh weights
    train_model(no_cancer_loader, num_epochs=1000, class_label=1)
    generate_synthetic(13, "NotCancerous", output_dir, class_label=1)

    # Optional: Save trained model
    # torch.save(unet.state_dict(), "unet_noncancerous_final.pth")
    # print("Non-cancerous model saved: unet_noncancerous_final.pth")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
