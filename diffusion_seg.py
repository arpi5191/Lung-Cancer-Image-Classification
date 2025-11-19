# Imports packages
import os
import gc
import torch
import shutil
import random
import tifffile
import numpy as np
from PIL import Image
import torch.nn as nn
from scipy.ndimage import zoom
from diffusers import UNet2DModel
from torchvision import transforms
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import AutoencoderKL

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
    Custom PyTorch Dataset for loading images of a single class.
    """
    def __init__(self, folder, transform=None):
        if isinstance(folder, list):
            self.files = folder
        else:
            self.files = []
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                if os.path.isdir(subfolder_path):
                    for f in os.listdir(subfolder_path):
                        if f.endswith(".tif"):
                            self.files.append(os.path.join(subfolder_path, f))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return idx, img

def load_dataset(base_dir):
    """
    Creates PyTorch DataLoaders for cancerous and non-cancerous image datasets.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    cancer_dir = os.path.join(base_dir, "Cancerous")
    no_cancer_dir = os.path.join(base_dir, "NotCancerous")

    # Reduced batch sizes for memory efficiency
    cancer_loader = DataLoader(SingleClassDataset(cancer_dir, transform),
                               batch_size=4, shuffle=True)
    no_cancer_loader = DataLoader(SingleClassDataset(no_cancer_dir, transform),
                                   batch_size=4, shuffle=True)

    return cancer_loader, no_cancer_loader

def load_model():
    """
    Initializes VAE, class-conditioned UNet, and scheduler for latent diffusion.

    Key improvement: UNet now accepts class labels for conditional generation.
    """
    global device, vae, unet, scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clear GPU memory before loading models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Load VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()  # Keep VAE in eval mode to save memory

    # Freeze VAE parameters (we don't train it)
    for param in vae.parameters():
        param.requires_grad = False

    # Load UNet with class conditioning
    unet = UNet2DModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        # CLASS CONDITIONING - This is the key addition
        class_embed_type="timestep",  # Use timestep-style embedding for classes
        num_class_embeds=2  # 2 classes: 0=cancerous, 1=non-cancerous
    ).to(device)

    # Initialize scheduler
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000
    )

    print(f"Models loaded on {device}")
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")

def train_model(dataloader, num_epochs=1000, class_label=0):
    """
    Train the UNet with class conditioning.

    Args:
        dataloader: PyTorch DataLoader with image batches
        num_epochs: Number of training epochs
        class_label: 0 for cancerous, 1 for non-cancerous
    """
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Use mixed precision for memory efficiency
    scaler = torch.cuda.amp.GradScaler()

    print(f"\nTraining class {class_label} ({'Cancerous' if class_label == 0 else 'Non-Cancerous'})")

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (indices, images) in enumerate(dataloader):
            images = images.to(device)

            # Encode images to latent space (no gradients for VAE)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215

            # Sample random timesteps
            t = torch.randint(0, scheduler.config.num_train_timesteps,
                            (latents.shape[0],), device=device).long()

            # Add noise to latents
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, t)

            # Create class labels for entire batch
            class_labels = torch.full((latents.shape[0],), class_label,
                                     dtype=torch.long, device=device)

            # Mixed precision training
            with torch.cuda.amp.autocast():
                # Predict noise with class conditioning
                noise_pred = unet(noisy_latents, t, class_labels=class_labels).sample
                loss = nn.MSELoss()(noise_pred, noise)

            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

            # Clear cache periodically
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

        scheduler_lr.step()
        avg_loss = epoch_loss / num_batches

        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler_lr.get_last_lr()[0]:.6f}")

        # Save checkpoint every 200 epochs
        if (epoch + 1) % 200 == 0:
            checkpoint_path = f"unet_class_{class_label}_epoch_{epoch+1}.pth"
            torch.save(unet.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

def generate_synthetic(num_images: int, classification, output_dir, class_label=0):
    """
    Generate synthetic images with class conditioning.

    Args:
        num_images: Number of images to generate
        classification: Folder name ("Cancerous" or "NotCancerous")
        output_dir: Base output directory
        class_label: 0 for cancerous, 1 for non-cancerous
    """
    img_size = 256
    target_size = 2048

    vae.eval()
    unet.eval()

    print(f"\nGenerating {num_images} {classification} images...")

    with torch.no_grad():
        for i in range(num_images):
            print(f"Generating image {i+1}/{num_images}...")

            # Sample random noise in latent space
            latents = torch.randn((1, 4, img_size//8, img_size//8)).to(device)

            # Create class label for conditioning
            class_labels = torch.full((1,), class_label, dtype=torch.long, device=device)

            # Set inference timesteps (more steps = better quality)
            scheduler.set_timesteps(50)

            # Iterative denoising with class conditioning
            for t in scheduler.timesteps:
                latent_model_input = scheduler.scale_model_input(latents, t)

                # Predict noise with class conditioning
                noise_pred = unet(latent_model_input, t, class_labels=class_labels).sample

                # Remove predicted noise
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Decode latent to pixel space
            image = vae.decode(latents / 0.18215).sample

            # Normalize to [0, 1]
            image = (image.clamp(-1, 1) + 1) / 2

            # Convert to grayscale
            image_np = image.squeeze(0).cpu().numpy()
            grayscale = image_np.mean(axis=0)

            # Upscale to target resolution
            if grayscale.shape != (target_size, target_size):
                zoom_factor = target_size / grayscale.shape[0]
                grayscale = zoom(grayscale, zoom_factor, order=3)

            # Convert to uint8
            grayscale = (grayscale * 255).astype(np.uint8)

            # Save image
            class_dir = os.path.join(output_dir, classification)
            image_name = f"generated_image_{i}"
            image_folder = os.path.join(class_dir, image_name)
            os.makedirs(image_folder, exist_ok=True)
            filepath = os.path.join(image_folder, f"{image_name}.tif")

            tifffile.imwrite(filepath, grayscale)
            print(f"Saved: {filepath}")

            # Clear cache
            torch.cuda.empty_cache()

def main():
    """
    Main training and generation pipeline with class conditioning.
    """
    # Setup output directory
    if os.path.exists('/.dockerenv'):
        output_dir = '/diffusion_tif'
    else:
        output_dir = '/ocean/projects/bio240001p/arpitha/diffusion_tif'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Directory '{output_dir}' deleted.")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory '{output_dir}' created.")

    # Create class subdirectories
    for label in ['Cancerous', 'NotCancerous']:
        path = os.path.join(output_dir, label)
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created.")

    # Setup base directory
    if os.path.exists('/.dockerenv'):
        base_dir = "/tumor_tif/"
    else:
        base_dir = '/ocean/projects/bio240001p/arpitha/tumor_tif'

    # Load datasets
    cancer_loader, no_cancer_loader = load_dataset(base_dir)
    print(f"\nCancerous dataset size: {len(cancer_loader.dataset)}")
    print(f"NotCancerous dataset size: {len(no_cancer_loader.dataset)}")

    # -------------------------------------------------------------------------
    # Train and generate CANCEROUS images (class_label=0)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 1: CANCEROUS IMAGES")
    print("="*60)

    load_model()
    train_model(cancer_loader, num_epochs=1000, class_label=0)
    generate_synthetic(21, "Cancerous", output_dir, class_label=0)

    # Save final model
    torch.save(unet.state_dict(), "unet_cancerous_final.pth")
    print("Cancerous model saved: unet_cancerous_final.pth")

    # -------------------------------------------------------------------------
    # Train and generate NON-CANCEROUS images (class_label=1)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 2: NON-CANCEROUS IMAGES")
    print("="*60)

    load_model()  # Reinitialize with fresh weights
    train_model(no_cancer_loader, num_epochs=1000, class_label=1)
    generate_synthetic(13, "NotCancerous", output_dir, class_label=1)

    # Save final model
    torch.save(unet.state_dict(), "unet_noncancerous_final.pth")
    print("Non-cancerous model saved: unet_noncancerous_final.pth")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
