# =============================================================================
# Synthetic Histopathology Image Generation
# Using Stable Diffusion + ControlNet (Canny) + DoRA Fine-Tuning
#
# Pipeline overview:
#   1. Load real FFPE lung histopathology images (Cancerous / NotCancerous)
#   2. Fine-tune DoRA adapters on the UNet for each class using the denoising objective
#   3. Generate synthetic images conditioned on class prompts and Canny edge maps
#
# DoRA (Weight-Decomposed Low-Rank Adaptation) adapts only a small fraction of
# UNet weights, keeping the base Stable Diffusion model frozen while learning
# domain-specific histopathology features (staining, nuclear morphology, gland architecture).
# =============================================================================

import os
import cv2
import time
import torch
import shutil
import random
import tifffile
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from torchvision import datasets, transforms
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline, DDPMScheduler

# =============================================================================
# Cache directories — redirected to project space to avoid HPC home quota errors.
# HuggingFace downloads models to ~/.cache by default, which fills up the small
# home directory quota on Bridges-2. These env vars redirect all downloads to
# the larger project storage space before any HF imports trigger a download.
# Must be set BEFORE any HuggingFace imports trigger model downloads.
# =============================================================================
os.environ['HF_HOME'] = '/ocean/projects/bio240001p/arpitha/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/ocean/projects/bio240001p/arpitha/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/ocean/projects/bio240001p/arpitha/hf_cache'
os.environ['TORCH_HOME'] = '/ocean/projects/bio240001p/arpitha/torch_cache'

# =============================================================================
# Global random seed — ensures reproducible image generation and training runs.
# Setting seeds in random, numpy, and torch (including CUDA) ensures that:
#   - DataLoader shuffling produces the same order every run
#   - Noise sampling in the diffusion process is identical across runs
#   - DoRA weight initializations are deterministic
# deterministic=True + benchmark=False forces cuDNN to use deterministic
# algorithms, at a small cost to speed.
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SingleClassDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for loading TIFF histopathology images of a single class.

    Supports two input modes:
        - Directory path: recursively collects all .tif files from subdirectories
        - List of file paths: uses the provided list directly

    The directory mode expects the following structure:
        folder/
            subfolder_1/image_1.tif
            subfolder_2/image_2.tif
            ...

    Args:
        folder (str or list): Path to folder containing .tif images, or explicit list of file paths.
        transform (callable, optional): Torchvision transforms to apply to each image.
    """
    def __init__(self, folder, transform=None):
        if isinstance(folder, list):
            # If a list of file paths is passed directly, use it as-is
            self.files = folder
        else:
            # Walk one level of subdirectories and collect all .tif files
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
        Load a single image, apply transforms, and return with metadata.

        Images are loaded as RGB to ensure consistent 3-channel format regardless
        of whether the source TIFF is grayscale or color. The filename is returned
        alongside the tensor to enable template -> generated traceability logging.

        Args:
            idx (int): Index of the image to load.

        Returns:
            tuple:
                idx (int): Index of the image (useful for debugging and logging).
                image_tensor (torch.Tensor): Transformed image tensor, shape [C, H, W].
                filename (str): Basename of the source file for provenance tracking.
        """
        img_path = self.files[idx]

        # Convert to RGB — ensures 3 channels even if source TIFF is grayscale.
        # Stable Diffusion's VAE expects 3-channel input.
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Return filename so we can log which real image each synthetic one was conditioned on
        filename = os.path.basename(img_path)
        return idx, img, filename


def load_dataset(base_dir):
    """
    Build PyTorch DataLoaders for Cancerous and NotCancerous histopathology images.

    Creates two sets of loaders:
        - Training loaders (batch_size=4, shuffle=True): for DoRA fine-tuning
        - Generation loaders (original batch sizes, shuffle=False): for ControlNet conditioning

    All images are resized to 512x512 and normalized to [-1, 1] to match the
    input format expected by Stable Diffusion's VAE and UNet.

    Args:
        base_dir (str): Root directory containing "Cancerous/" and "NotCancerous/" subfolders.

    Returns:
        tuple: (cancer_loader_train, no_cancer_loader_train, cancer_loader_gen, no_cancer_loader_gen)
    """
    # Normalize to [-1, 1]: ToTensor() maps [0,255] -> [0,1], then Normalize([0.5],[0.5])
    # maps [0,1] -> [-1,1] via (x - 0.5) / 0.5. This matches SD's expected input range.
    transform = transforms.Compose([
        transforms.Resize((512, 512)),       # SD operates at 512x512
        transforms.ToTensor(),               # [0,255] uint8 HWC -> [0,1] float CHW
        transforms.Normalize([0.5], [0.5])   # [0,1] -> [-1,1]
    ])

    cancer_dir    = os.path.join(base_dir, "Cancerous")
    no_cancer_dir = os.path.join(base_dir, "NotCancerous")

    # Training loaders: small batch size, shuffled — good for gradient diversity during DoRA training
    # Generation loaders: full dataset in one batch, unshuffled — used to extract Canny conditioning images
    cancer_loader_train    = DataLoader(SingleClassDataset(cancer_dir, transform),    batch_size=4,  shuffle=True)
    no_cancer_loader_train = DataLoader(SingleClassDataset(no_cancer_dir, transform), batch_size=4,  shuffle=True)
    cancer_loader_gen      = DataLoader(SingleClassDataset(cancer_dir, transform),    batch_size=21, shuffle=False)
    no_cancer_loader_gen   = DataLoader(SingleClassDataset(no_cancer_dir, transform), batch_size=13, shuffle=False)

    return cancer_loader_train, no_cancer_loader_train, cancer_loader_gen, no_cancer_loader_gen


def load_model():
    """
    Load the Stable Diffusion + Canny ControlNet pipeline.

    Uses 'Nihirc/Prompt2MedImage' (SD fine-tuned on medical imaging) paired with
    'lllyasviel/sd-controlnet-canny'. Canny edge maps extracted from real histopathology
    images provide structural conditioning, preserving gland boundaries and nuclear layout.

    Why Canny ControlNet:
        - sd-controlnet-canny expects binary edge maps as input, which we generate from
          real histopathology images using OpenCV's Canny detector
        - This gives the model structural guidance (gland outlines, nuclear boundaries)
          without requiring segmentation masks or depth maps
        - Other ControlNet variants (seg, depth) require different input formats and
          produce black images when fed plain histopathology images

    DoRA adapter injection is handled separately in finetune_dora().

    Returns:
        pipe (StableDiffusionControlNetPipeline): Loaded pipeline, ready for DoRA injection.
        device (str): 'cuda' or 'cpu'.
    """
    model_id      = "Nihirc/Prompt2MedImage"
    controlnet_id = "lllyasviel/sd-controlnet-canny"
    # controlnet_id = "lllyasviel/sd-controlnet-depth"  # requires depth maps as input
    # controlnet_id = "lllyasviel/sd-controlnet-seg"    # requires ADE20K segmentation colour maps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading Medical Stable Diffusion model on {device} with dtype={dtype}...")

    # Load ControlNet weights separately so they can be passed into the SD pipeline
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)

    # Load the full SD pipeline with ControlNet attached.
    # safety_checker=None disables the NSFW filter — not needed for medical images
    # and saves VRAM on the HPC GPU.
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None
    )

    pipe = pipe.to(device)

    if device == "cuda":
        # Attention slicing reduces VRAM usage by computing attention one slice at a time
        # instead of all heads simultaneously. Slight speed tradeoff, necessary on 40GB A100
        # when also holding the merged UNet in memory.
        pipe.enable_attention_slicing()
        print("GPU optimizations enabled (attention slicing)")

    print("Model loaded successfully.\n")
    return pipe, device


def finetune_dora(base_pipe, device, dtype, train_loaders, classifications, prompts, output_dir,
                  num_epochs=5, lr=1e-4):
    """
    Fine-tune class-specific DoRA adapters on real histopathology images.

    IMPORTANT — this function takes a plain StableDiffusionPipeline (NO ControlNet).
    Passing a StableDiffusionControlNetPipeline causes two silent failures:
        1. pipe.unet() called directly skips ControlNet residuals -> wrong forward pass
           -> bad gradients -> corrupted adapter weights -> black images at generation
        2. The returned PeftModel wrapper drops ControlNet kwargs at inference
           -> black images even if training was correct

    Only DoRA adapter weights are updated. All base SD weights (VAE, text encoder,
    UNet backbone) remain frozen throughout.

    Training objective (epsilon prediction / DDPM denoising loss):
        1. Encode real images into latent space via frozen VAE
        2. Sample random noise + timestep, add noise to latents
        3. DoRA-adapted UNet predicts the noise
        4. MSE loss between predicted and actual noise
        5. Backprop through DoRA adapter weights only

    Each class gets an independently trained adapter injected fresh from the clean
    base UNet, preventing cross-class weight contamination.

    After all classes are trained, merge_and_unload() folds the DoRA adapter deltas
    into the base weight matrices and strips the PEFT wrapper entirely, returning a
    plain UNet2DConditionModel. This is required so that when the UNet is placed into
    a ControlNet pipeline, the ControlNet residual kwargs are forwarded correctly
    rather than being silently dropped by the PEFT wrapper.

    Args:
        base_pipe (StableDiffusionPipeline): Plain SD pipeline — NO ControlNet.
        device (str): 'cuda' or 'cpu'.
        dtype (torch.dtype): float16 on CUDA, float32 on CPU. Must match base_pipe.
        train_loaders (list[DataLoader]): Per-class loaders, batch_size=4, shuffle=True.
        classifications (list[str]): e.g. ['Cancerous', 'NotCancerous'].
        prompts (list[str]): Positive text prompt per class for UNet conditioning.
        output_dir (str): Where to save trained adapter weights per class.
        num_epochs (int): Epochs per class. Increase if loss hasn't converged.
        lr (float): AdamW learning rate. 1e-4 is standard for LoRA/DoRA.

    Returns:
        merged_unet (UNet2DConditionModel): Plain UNet with DoRA weights baked in,
            in eval() mode, cast to dtype.
    """
    print(f"\n{'='*60}")
    print("DoRA fine-tuning  (plain SD UNet — no ControlNet during training)")
    print(f"Model dtype: {dtype}")
    print(f"{'='*60}")

    # Load the DDPM noise scheduler from the same model checkpoint.
    # This defines how noise is added at each timestep (forward diffusion process)
    # and is needed to compute: noisy_latents = add_noise(clean_latents, noise, t)
    noise_scheduler = DDPMScheduler.from_pretrained(
        "Nihirc/Prompt2MedImage", subfolder="scheduler"
    )

    # DoRA configuration:
    #   use_dora=True: decomposes each adapted weight into a magnitude scalar and a
    #       direction matrix, giving more expressive adaptation than plain LoRA
    #   r=16: rank of the low-rank adapter matrices. Higher rank = more parameters
    #       = more expressive, but more VRAM. r=16 is a good balance for small datasets.
    #   lora_alpha=64: scaling factor. Effective scale = alpha/r = 64/16 = 4x.
    #       Higher scale = adapter has stronger influence over the base weights.
    #   target_modules: the four linear projection layers inside each UNet attention block.
    #       to_q/to_k/to_v = query/key/value projections; to_out.0 = output projection.
    #       These are the most impactful layers for style and domain adaptation.
    #   lora_dropout=0.1: regularizes the small dataset by randomly zeroing adapter activations
    #   task_type="TEXT_TO_IMAGE": tells PEFT this is a diffusion UNet, not a language model
    config = LoraConfig(
        use_dora=True,
        r=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type="TEXT_TO_IMAGE"
    )

    # Freeze VAE and text encoder — only UNet adapter weights should receive gradients.
    # requires_grad_(False) detaches all parameters from the computation graph so no
    # gradients are computed for them and they consume no optimizer memory.
    base_pipe.vae.requires_grad_(False)
    base_pipe.text_encoder.requires_grad_(False)

    # -----------------------------------------------------------------------
    # GradScaler for float16 mixed precision training.
    #
    # WHY THIS IS NEEDED:
    # float16 has a max value of ~65504. During the UNet forward pass, intermediate
    # activations in attention layers can exceed this, becoming inf, which propagates
    # as NaN through the loss and all gradients. Without a scaler, every epoch
    # produces NaN loss and the adapter weights are permanently corrupted from batch 1.
    #
    # GradScaler works by:
    #   1. Multiplying the loss by a large scale factor before backward(), keeping
    #      gradients large enough to be representable in float16
    #   2. Dividing back (unscaling) before optimizer.step() so weight updates are correct
    #   3. Skipping the step entirely if any gradient is still inf/NaN after unscaling
    #   4. Automatically adjusting the scale factor up/down each step
    # -----------------------------------------------------------------------
    use_amp = (device == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    for class_idx, (loader, prompt) in enumerate(zip(train_loaders, prompts)):
        print(f"\nFine-tuning DoRA for class: {classifications[class_idx]}")
        print(f"Dataset size: {len(loader.dataset)} images | Epochs: {num_epochs}")

        # ------------------------------------------------------------------
        # Unwrap any previous PEFT wrapper to recover the clean base UNet,
        # then inject a fresh set of DoRA adapters for this class.
        #
        # WHY: After the first class trains, base_pipe.unet is a PeftModel wrapping
        # the base UNet. For the second class we need a clean UNet so Cancerous
        # adapter weights don't leak into NotCancerous training.
        # base_model.model gives us the raw UNet2DConditionModel back.
        # ------------------------------------------------------------------
        raw_unet = (base_pipe.unet.base_model.model
                    if hasattr(base_pipe.unet, 'base_model')
                    else base_pipe.unet)
        base_pipe.unet = get_peft_model(raw_unet, config)

        # ------------------------------------------------------------------
        # Cast ONLY the trainable DoRA adapter parameters to float32.
        #
        # WHY: GradScaler.unscale_() requires parameters being optimized to be
        # float32. It cannot unscale float16 gradients and raises:
        #   ValueError: Attempting to unscale FP16 gradients
        # The frozen base UNet weights stay float16 (VRAM efficient).
        # The forward pass still runs in float16 via autocast.
        # Only the adapter weight updates happen in float32 (numerically stable).
        # ------------------------------------------------------------------
        for name, param in base_pipe.unet.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()  # float32 for trainable adapter params only

        base_pipe.unet.train()

        # Optimizer must be created AFTER get_peft_model() and the float32 cast,
        # so it points to the correct newly injected float32 adapter parameters.
        # Reusing an old optimizer would reference stale params from the previous class.
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, base_pipe.unet.parameters()),
            lr=lr
        )

        # Encode the class prompt once and reuse for every batch.
        # The text encoder is frozen so the embedding never changes.
        text_inputs = base_pipe.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            # encoder_hidden_states shape: [1, seq_len, 768]
            # This is the cross-attention conditioning signal passed to the UNet
            encoder_hidden_states = base_pipe.text_encoder(**text_inputs)[0]

        for epoch in range(num_epochs):
            epoch_loss  = 0.0
            num_batches = 0

            for _, batch_images, _ in loader:
                # Cast images to float16 to match the VAE weights.
                # The VAE was loaded with torch_dtype=float16 so its conv weights
                # are float16 — input must match or a dtype mismatch error is raised.
                batch_images = batch_images.to(device=device, dtype=dtype)

                with torch.no_grad():
                    # Step 1: Encode images into the latent space via the frozen VAE.
                    # The VAE compresses 512x512x3 images -> 64x64x4 latents.
                    # scaling_factor (~0.18) normalizes latents to unit variance,
                    # which is what the UNet expects as input.
                    latents = base_pipe.vae.encode(batch_images).latent_dist.sample()
                    latents = latents * base_pipe.vae.config.scaling_factor
                    latents = latents.to(dtype=dtype)  # ensure float16 after VAE encode

                # Step 2: Sample random Gaussian noise with the same shape as the latents
                noise = torch.randn_like(latents)

                # Step 3: Sample a random timestep for each image in the batch.
                # The DDPM scheduler has 1000 timesteps (0=clean, 999=pure noise).
                # Training on random timesteps teaches the UNet to denoise at any
                # noise level — this is the standard diffusion training objective.
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()

                # Step 4: Forward diffusion — add noise to clean latents at the sampled
                # timestep. This is the corrupted input the UNet will try to denoise.
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Step 5: UNet forward pass under autocast.
                # autocast selects float16 for safe ops and float32 for numerically
                # sensitive ones (e.g. softmax in attention), preventing NaN.
                # The UNet predicts the noise that was added (epsilon prediction objective).
                with torch.autocast(device_type=device, enabled=use_amp):
                    noise_pred = base_pipe.unet(
                        noisy_latents,
                        timesteps,
                        # Expand prompt embedding from [1, seq, 768] to [batch, seq, 768]
                        encoder_hidden_states=encoder_hidden_states.expand(
                            latents.shape[0], -1, -1
                        )
                    ).sample

                # Step 6: Epsilon prediction loss — MSE between predicted and actual noise.
                # Computed in float32 to avoid NaN from float16 MSE overflow.
                loss = F.mse_loss(noise_pred.float(), noise.float())

                optimizer.zero_grad()

                # Step 7: Scaled backward pass.
                # scaler.scale(loss) multiplies by the current scale factor before
                # backward() so gradients stay in float16's representable range.
                scaler.scale(loss).backward()

                # Unscale gradients back to their true magnitude before clipping/stepping.
                # Must be called before clip_grad_norm_ and optimizer.step().
                scaler.unscale_(optimizer)

                # Step 8: Gradient clipping — caps gradient norm at 1.0.
                # With only 21/13 training images, a single outlier batch can produce
                # very large gradients. Clipping rescales the gradient vector if its
                # norm exceeds 1.0, protecting adapter weights from corruption.
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, base_pipe.unet.parameters()),
                    max_norm=1.0
                )

                # scaler.step() skips the update if any gradient is still inf/NaN,
                # protecting adapter weights from corruption on bad batches.
                scaler.step(optimizer)

                # scaler.update() adjusts the scale factor for the next iteration:
                # increases if no inf/NaN found, decreases if they were.
                scaler.update()

                loss_val = loss.item()
                if not torch.isfinite(torch.tensor(loss_val)):
                    print(f"    WARNING: non-finite loss {loss_val} — skipping batch")
                    continue

                epoch_loss  += loss_val
                num_batches += 1

            if num_batches > 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] | Avg Loss: {epoch_loss/num_batches:.4f}")
            else:
                print(f"  Epoch [{epoch+1}/{num_epochs}] | All batches produced NaN — check data")

        # Save the per-class adapter weights to disk.
        # These can be reloaded with PeftModel.from_pretrained() to skip retraining.
        adapter_path = os.path.join(output_dir, f"dora_adapter_{classifications[class_idx]}")
        base_pipe.unet.save_pretrained(adapter_path)
        print(f"\nDoRA adapter saved: {adapter_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Merge DoRA adapter deltas into base weights and strip PEFT wrapper.
    #
    # WHY merge_and_unload() is required:
    # base_pipe.unet is currently a PeftModel wrapping the UNet. If transplanted
    # into the ControlNet pipeline as-is, the PEFT wrapper intercepts forward()
    # and silently drops the ControlNet residual kwargs
    # (down_block_additional_residuals, mid_block_additional_residual).
    # Denoising then runs without ControlNet conditioning -> black images.
    #
    # merge_and_unload() folds the DoRA magnitude + direction components directly
    # into the base weight matrices, returning a plain UNet2DConditionModel that
    # forwards all kwargs correctly. All learned adaptations are preserved —
    # they are now baked into the weights rather than sitting in a separate adapter.
    #
    # Cast back to dtype immediately: merge arithmetic upcasts to float32 internally.
    # ------------------------------------------------------------------
    print("\nMerging DoRA weights into base UNet (stripping PEFT wrapper)...")
    merged_unet = base_pipe.unet.merge_and_unload()
    merged_unet = merged_unet.to(dtype=dtype)
    merged_unet.eval()
    print(f"Merge complete — type: {type(merged_unet).__name__}  "
          f"dtype: {next(merged_unet.parameters()).dtype}")

    print(f"\n{'='*60}")
    print("DoRA fine-tuning complete.")
    print(f"{'='*60}\n")

    return merged_unet


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
        - ControlNet uses a Canny edge map of a real image as structural guidance.
        - Each generated image is conditioned on one real image's edge map.
    """
    # Initialize the total latency
    total_latency = 0

    for class_idx in range(len(classifications)):
        print(f"\n{'='*60}")
        print(f"Generating {num_images[class_idx]} images for class: {classifications[class_idx]}")
        print(f"{'='*60}")

        # Compute and log the text embedding shape for verification.
        # The pipeline uses this internally — we compute it here only for logging.
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

        # Create an iterator over the generation DataLoader.
        # We cycle back to the start if we run out of conditioning images.
        data_iter = iter(loaders[class_idx])

        for img_idx in range(num_images[class_idx]):
            print(f"\n[{img_idx+1}/{num_images[class_idx]}] Generating image...")

            # -----------------------------------------
            # Retrieve a batch of real images for ControlNet conditioning.
            # StopIteration is caught to cycle through the dataset if num_images
            # exceeds the dataset size.
            # -----------------------------------------
            try:
                _, batch_images, batch_filenames = next(data_iter)
            except StopIteration:
                data_iter = iter(loaders[class_idx])
                _, batch_images, batch_filenames = next(data_iter)

            # -----------------------------------------
            # Select one image from the batch as the ControlNet conditioning image.
            # img_idx % batch_size cycles through images in the batch before
            # fetching the next batch, ensuring a different real image each time.
            # -----------------------------------------
            batch_size = batch_images.shape[0]
            j = img_idx % batch_size

            conditioning_image    = batch_images[j]
            conditioning_filename = batch_filenames[j]

            # -----------------------------------------
            # Build the Canny edge map for ControlNet conditioning.
            #
            # WHY Canny edges:
            #   sd-controlnet-canny expects a binary edge map, not a raw photo.
            #   Canny edges capture gland boundaries and nuclear outlines —
            #   the structural features we want preserved in synthetic images.
            #
            # Steps:
            #   1. Undo [-1,1] normalization -> [0,1], clamp for safety
            #   2. Convert tensor [C,H,W] -> numpy [H,W,C] uint8 for OpenCV
            #   3. Convert RGB -> grayscale (Canny operates on single channel)
            #   4. Apply Canny (thresholds 100/200 are standard SD defaults)
            #   5. Stack edges into 3-channel RGB PIL — pipeline expects RGB PIL input
            # -----------------------------------------
            conditioning_image = ((conditioning_image + 1) / 2).clamp(0, 1)
            conditioning_np    = (conditioning_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            gray               = cv2.cvtColor(conditioning_np, cv2.COLOR_RGB2GRAY)
            edges              = cv2.Canny(gray, 100, 200)
            conditioning_image = Image.fromarray(np.stack([edges] * 3, axis=-1))
            conditioning_image = conditioning_image.resize((512, 512))

            # Different seed per image for variety, but reproducible across runs
            generator = torch.Generator(device=device).manual_seed(42 + img_idx)

            # Start timing per-image generation
            # Ensure all previous GPU ops are finished
            torch.cuda.synchronize()
            start = time.time()

            # -----------------------------------------
            # Generate the synthetic image.
            #
            # Key parameters:
            #   controlnet_conditioning_scale=0.8: how strongly the Canny edge map
            #       guides output structure. Values >1.0 collapse output to black.
            #   num_inference_steps=50: denoising steps. Standard SD default.
            #   guidance_scale=7.5: classifier-free guidance strength. Higher = output
            #       follows the prompt more closely. 7.5 is the SD default.
            # -----------------------------------------
            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(device == "cuda")):
                    result = pipe(
                        prompt=prompts[class_idx],
                        negative_prompt=negative_prompts[class_idx],
                        image=conditioning_image,
                        controlnet_conditioning_scale=0.8,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        width=512,
                        height=512,
                        generator=generator
                    )

            # Wait for all GPU ops in this pipeline call to finish
            torch.cuda.synchronize()
            end = time.time()

            # Compute latency
            latency = end - start                # seconds per image
            total_latency += latency

            image = result.images[0]

            # Pixel stats to verify output is not black (mean ~0) or white (mean ~255).
            # A healthy histopathology image should have mean roughly in 50-200.
            arr = np.array(image)
            print(f"  pixel stats — min:{arr.min()} max:{arr.max()} mean:{arr.mean():.1f}")

            # Upsample 512x512 -> 2048x2048 to match original TIFF resolution.
            # LANCZOS is the highest quality resampling filter available in PIL.
            image = image.resize((2048, 2048), Image.Resampling.LANCZOS)

            # Convert to 8-bit grayscale.
            # convert("L") = 8-bit grayscale uint8  <- CORRECT
            # convert("1") = 1-bit binary           <- WRONG, causes blank white TIFFs
            image = image.convert("L")

            # -----------------------------------------
            # Save as TIFF with explicit uint8 dtype.
            # Each image gets its own subfolder mirroring the input dataset structure
            # so downstream classifiers expecting that layout work without modification.
            # -----------------------------------------
            class_dir    = os.path.join(output_dir, classifications[class_idx])
            image_name   = f"context_image_{img_idx}"
            image_folder = os.path.join(class_dir, image_name)
            os.makedirs(image_folder, exist_ok=True)
            filepath = os.path.join(image_folder, f"{image_name}.tif")

            image_array = np.array(image, dtype=np.uint8)
            tifffile.imwrite(filepath, image_array)
            print(f"Saved to: {filepath}")
            print(f"Template -> Generated | {conditioning_filename} -> {image_name}.tif")

            # Free GPU memory after each image — important when generating many images
            # sequentially on a shared HPC node
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Compute and print the average throughput across all generated images
    # Throughput is defined as images processed per second
    # total_latency is the sum of all per-batch or per-image latencies
    print()
    average_throughput = sum(num_images) / total_latency
    print(f"Average Throughput: {average_throughput:.6f} images/sec")
    print()

    print(f"\n{'='*60}")
    print(f"All {sum(num_images)} images saved to {output_dir}")
    print(f"{'='*60}")


def main():
    """
    Orchestrates the full synthetic histopathology image generation pipeline.

    Steps:
        1. Create output directories for generated images
        2. Load DataLoaders for training and generation datasets per class
        3. Define prompts and negative prompts for each class
        4. Load a plain Stable Diffusion pipeline for DoRA fine-tuning
        5. Validate prompt token lengths against CLIP's 77-token limit
        6. Fine-tune DoRA adapters per class on real histopathology images
        7. Build a ControlNet pipeline and replace its UNet with the merged DoRA UNet
        8. Generate synthetic images per class using the fine-tuned pipeline
    """

    # =============================================================================
    # Step 0: Set directories based on environment.
    # Docker path is for local testing; the else branch is for Bridges-2 HPC.
    # =============================================================================
    if os.path.exists('/.dockerenv'):
        output_dir = '/context_tif'
        base_dir   = '/tif/'
    else:
        output_dir = '/ocean/projects/bio240001p/arpitha/context_tif'
        base_dir   = '/ocean/projects/bio240001p/arpitha/tumor_tif'

    # Delete and recreate output directory to avoid mixing images from different runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Directory '{output_dir}' deleted.")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory '{output_dir}' created.")

    # Create per-class subdirectories mirroring the input dataset structure
    for label in ['Cancerous', 'NotCancerous']:
        path = os.path.join(output_dir, label)
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created.")

    # =============================================================================
    # Step 1: Load dataset
    # =============================================================================
    cancer_loader_train, no_cancer_loader_train, cancer_loader_gen, no_cancer_loader_gen = load_dataset(base_dir)

    print(f"Cancerous train:    {len(cancer_loader_train.dataset)} images")
    print(f"NotCancerous train: {len(no_cancer_loader_train.dataset)} images")
    print(f"Cancerous gen:      {len(cancer_loader_gen.dataset)} images")
    print(f"NotCancerous gen:   {len(no_cancer_loader_gen.dataset)} images")

    # train_loaders: used in finetune_dora() for the denoising training objective
    # gen_loaders:   used in train_model() as the source of Canny conditioning images
    train_loaders = [cancer_loader_train, no_cancer_loader_train]
    gen_loaders   = [cancer_loader_gen,   no_cancer_loader_gen]

    num_images      = [21, 13]   # number of synthetic images to generate per class
    classifications = ["Cancerous", "NotCancerous"]

    # =============================================================================
    # Step 2: Define prompts and negative prompts.
    #
    # Prompts guide the diffusion process toward the desired appearance.
    # Negative prompts steer away from unwanted features via classifier-free guidance.
    #
    # Structure:
    #   base_prompt:            shared anatomical context for both classes
    #   class-specific prompt:  distinguishing morphological features per class
    #   base_negative:          shared artifacts to avoid in all outputs
    #   class-specific negative: features that would make one class look like the other
    # =============================================================================
    base_prompt = "FFPE lung adenocarcinoma grayscale microscopy from a male, Acinar architecture, "

    malignant_prompt = (
        "Malignant cells in irregular glands, "
        "Most images show tangential epithelial sections, "
        "Rarely, images have branching tubular structures with open lumens, nuclei localized along the lumen periphery, "
        "Enlarged pleomorphic hyperchromatic nuclei, "
        "High N/C ratio"
    )

    benign_prompt = (
        "Benign cells in intact glands, "
        "Branching tubular structures with open lumens, nuclei localized along the lumen periphery, "
        "Small uniform normochromatic nuclei, "
        "Normal N/C ratio"
    )

    prompts = [
        base_prompt + malignant_prompt,  # Cancerous
        base_prompt + benign_prompt      # NotCancerous
    ]

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
        base_negative + malignant_avoid,  # Cancerous
        base_negative + benign_avoid      # NotCancerous
    ]

    all_prompts = prompts + negative_prompts

    # =============================================================================
    # Step 3: Determine device and dtype.
    # float16 on CUDA halves VRAM usage with minimal quality impact.
    # float32 on CPU is required as CPU float16 ops are slow/unsupported.
    # =============================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    # =============================================================================
    # Step 4: Load plain Stable Diffusion pipeline (no ControlNet) for DoRA training.
    #
    # WHY no ControlNet here:
    #   During training we call pipe.unet() directly. If ControlNet were attached,
    #   this direct call skips the ControlNet residuals entirely, producing wrong
    #   gradients that corrupt the DoRA adapter weights -> black images at generation.
    #   ControlNet is only needed at generation time (Step 7).
    # =============================================================================
    base_pipe = StableDiffusionPipeline.from_pretrained(
        "Nihirc/Prompt2MedImage",
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)

    tokenizer = base_pipe.tokenizer

    # =============================================================================
    # Step 5: Prompt Token Validation.
    # CLIP silently truncates prompts at 77 tokens. Any content beyond token 77
    # is ignored. Log a warning if any prompt hits the limit so it can be shortened.
    # =============================================================================
    expected_tokens = 77

    for i, prompt in enumerate(all_prompts):
        text_inputs    = tokenizer(prompt, truncation=True, return_tensors="pt")
        num_tokens     = text_inputs.input_ids.shape[1]
        tokens_decoded = [tokenizer.decode([tid]).strip() for tid in text_inputs.input_ids[0]]

        if num_tokens >= expected_tokens:
            print(f"WARNING: Prompt {i} hit the {expected_tokens}-token limit and may be truncated.")

        print(f"Prompt {i}: {num_tokens} tokens")
        print(tokens_decoded)
        print()

    # =============================================================================
    # Step 6: Fine-tune DoRA adapters on real histopathology images.
    # Returns a merged UNet2DConditionModel with DoRA weights baked in.
    # =============================================================================
    merged_unet = finetune_dora(
        base_pipe       = base_pipe,
        device          = device,
        dtype           = dtype,
        train_loaders   = train_loaders,
        classifications = classifications,
        prompts         = prompts,
        output_dir      = output_dir,
        num_epochs      = 30,
        lr              = 1e-4
    )

    # Free the plain SD pipeline from GPU memory — only the merged UNet is needed now
    del base_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =============================================================================
    # Step 7: Build ControlNet pipeline and swap in the merged DoRA UNet.
    #
    # We load a fresh ControlNet pipeline (with its original UNet), then replace
    # its UNet with the DoRA-merged one. This gives us ControlNet conditioning at
    # generation time while using the histopathology-adapted weights from training.
    # =============================================================================
    pipe, _ = load_model()
    pipe.unet = merged_unet.to(device=device, dtype=dtype)
    pipe.unet.eval()  # disable dropout, use inference-mode behaviour

    # Sanity checks — confirm all components are on the correct device and dtype
    print(f"UNet type:  {type(pipe.unet).__name__}")            # must be UNet2DConditionModel, NOT PeftModel
    print(f"UNet dtype: {next(pipe.unet.parameters()).dtype}")   # must be torch.float16
    print(f"UNet device:{next(pipe.unet.parameters()).device}")  # must be cuda:0
    print(f"ControlNet dtype: {next(pipe.controlnet.parameters()).dtype}")  # must match UNet dtype

    # =============================================================================
    # Step 8: Generate synthetic images using the fine-tuned ControlNet pipeline.
    # =============================================================================
    train_model(
        pipe, device, output_dir, classifications,
        prompts, negative_prompts, num_images, gen_loaders
    )

if __name__ == "__main__":
    main()
