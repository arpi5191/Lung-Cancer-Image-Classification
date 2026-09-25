# Import packages
# Core Python
import os
import time
import random
import shutil
from pathlib import Path
from collections import Counter
import argparse
import copy

# Scientific / Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Image handling
from PIL import Image
import tifffile

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

# Torchvision
import torchvision
from torchvision import transforms, datasets, models

# Scikit-learn metrics
from sklearn.metrics import precision_score, recall_score, f1_score

# Define the global variables for feature dimension, number of classes, and activation
global feature_dim
global num_classes
global activation

# Spatial size (height = width) of each input image patch in pixels.
# ResNet18 with AdaptiveAvgPool2d(1,1) can accept any spatial size, so this
# value is also used when reshaping tensors in the train/eval loops.
feature_dim = 512

# Number of output classes for the final classification layer (e.g. Cancerous vs NotCancerous).
num_classes = 2

# Shared dictionary populated by forward hooks; maps layer name → detached output tensor.
# Populated during each forward pass so embeddings can be retrieved outside the model.
activation = {}

def get_patch_files(patches_dir, image_patches, total_image_count):
    """
    Collect image patch file paths from every subdirectory inside `patches_dir`.

    Iterates one level deep: each immediate child directory of `patches_dir` is
    treated as a "patch group" whose image files are gathered into a list.
    System / metadata files (e.g. `.DS_Store`) are excluded.

    Args:
        patches_dir (str): Root directory whose subdirectories contain image patches.
        image_patches (dict): Accumulator mapping subdirectory path → list of image
            file paths. Updated in-place and also returned.
        total_image_count (int): Running total of image files seen so far across
            all previous calls. Incremented by the number of files found here.

    Returns:
        tuple:
            image_patches (dict): Updated mapping of subdirectory → image file list.
            total_image_count (int): Updated cumulative file count.
    """

    # Iterate through each item in the specified patches directory
    for dir_name in os.listdir(patches_dir):

        # Print the current directory name for tracking progress
        print(f"Processing directory: {dir_name}")

        # Construct the full absolute path to the current directory
        dir_path = os.path.join(patches_dir, dir_name)
        print(f"Full path: {dir_path}")

        # Only continue if this path is a directory (skip loose files)
        if not os.path.isdir(dir_path):
            continue

        # List all image files in this directory, ignoring system files such as .DS_Store
        image_files = [
            os.path.join(dir_path, file)
            for file in os.listdir(dir_path)
            if file != '.DS_Store' and os.path.isfile(os.path.join(dir_path, file))
        ]

        # Store the list of image files in the dictionary under this directory's path
        image_patches[dir_path] = image_files

        # Increment the total count of image files by the number found here
        total_image_count += len(image_files)

    # Return the updated dictionary and the total image count
    return image_patches, total_image_count


def create_data_directories():
    """
    Build a clean directory tree for train / validation / test datasets.

    Detects whether the script is running inside a Docker container and
    chooses the appropriate base path. Any pre-existing directory at that
    path is deleted before the new structure is created, ensuring a clean
    slate for each run.

    Directory layout created:
        <data_dir>/
            train/
                Cancerous/
                NotCancerous/
            val/
                Cancerous/
                NotCancerous/
            test/
                Cancerous/
                NotCancerous/

    Returns:
        tuple[str, str, str, str]:
            data_dir  – Base data directory.
            train_dir – Training split directory.
            val_dir   – Validation split directory.
            test_dir  – Test split directory.
    """

    # Detect environment and set base data directory path accordingly
    if os.path.exists('/.dockerenv'):
        data_dir = '/data'  # Docker environment base directory
    else:
        # data_dir = "/Users/arpitha/Documents/Research_Labs/Lab_Schwartz/Lung-Cancer-Image-Classification-main/data"
        data_dir = '/ocean/projects/bio240001p/arpitha/data'  # Local environment

    # Delete existing base directory if it exists to start fresh
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print(f"Directory '{data_dir}' has been deleted.")
    else:
        print(f"Directory '{data_dir}' does not exist.")

    # Create base data directory
    os.makedirs(data_dir, exist_ok=True)
    print(f"Directory '{data_dir}' created successfully." if os.path.exists(data_dir) else f"Failed to create '{data_dir}'.")

    def create_subdirs(base_path):
        """
        Create `base_path` and the two class subdirectories beneath it.

        Args:
            base_path (str): Path for a dataset split (train / val / test).
        """
        os.makedirs(base_path, exist_ok=True)
        for subfolder in ['Cancerous', 'NotCancerous']:
            path = os.path.join(base_path, subfolder)
            os.makedirs(path, exist_ok=True)
            print(f"Directory '{path}' created successfully." if os.path.exists(path) else f"Failed to create '{path}'.")

    # Define and create train directories
    train_dir = os.path.join(data_dir, 'train')
    create_subdirs(train_dir)

    # Define and create validation directories
    val_dir = os.path.join(data_dir, 'val')
    create_subdirs(val_dir)

    # Define and create test directories
    test_dir = os.path.join(data_dir, 'test')
    create_subdirs(test_dir)

    # Return paths for base, train, val, and test directories
    return data_dir, train_dir, val_dir, test_dir


def copy_directories_to_directories(dst_dir, dirs):
    """
    Copy patch group directories into the appropriate class subfolder of `dst_dir`.

    Each directory in `dirs` is inspected for the substring ``"Cancerous"`` or
    ``"NotCancerous"`` in its path. Matching directories are copied (via
    ``shutil.copytree``) under the corresponding class subfolder inside
    `dst_dir`. Directories whose names match neither category are silently
    skipped.

    Args:
        dst_dir (str): Destination split root (e.g. the train or val directory).
        dirs (list[str]): Source directory paths to distribute.
    """

    # Target subdirectories for each class within the destination split root
    cancer_dir_path = os.path.join(dst_dir, 'Cancerous')
    no_cancer_dir_path = os.path.join(dst_dir, 'NotCancerous')

    # Iterate over source directories
    for dir in dirs:

        source_dir = Path(dir)

        # Route to the correct class subfolder based on the directory name
        if "NotCancerous" in str(dir):
            target_dir = no_cancer_dir_path
        elif "Cancerous" in str(dir):
            target_dir = cancer_dir_path
        else:
            # Skip directories that don't match either expected category
            continue

        # Compose full target path including source directory name
        target_path = os.path.join(target_dir, os.path.basename(source_dir))

        # Copy the entire source directory tree to the target location
        shutil.copytree(source_dir, target_path)

        print(f"Directory '{source_dir}' has been copied into '{target_dir}' as '{target_path}'.")


def define_train_transformer():
    """
    Build the image transformation pipeline used for training data.

    Augmentations are applied to improve generalisation:
    - Grayscale conversion (1 channel) to match the single-channel model input.
    - Random horizontal and vertical flips for spatial augmentation.
    - Random brightness jitter (±150 %) while leaving contrast, saturation,
      and hue unchanged (no effect for grayscale images).
    - Conversion to a PyTorch ``FloatTensor`` scaled to [0, 1].

    Returns:
        torchvision.transforms.Compose: Composed transform for training images.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to single-channel grayscale
        transforms.RandomHorizontalFlip(),            # Random horizontal flip (50 % chance)
        transforms.RandomVerticalFlip(),              # Random vertical flip  (50 % chance)
        transforms.ColorJitter(
            brightness=1.5,    # Brightness varies up to ±150 %
            contrast=1.0,      # Contrast factor of 1.0 → no change
            saturation=1.0,    # Saturation factor of 1.0 → no change (also irrelevant for grayscale)
            hue=0              # No hue adjustment
        ),
        transforms.ToTensor(),                        # Convert PIL image to FloatTensor in [0, 1]
        # transforms.Lambda(lambda x: x.reshape(1, 4608, 4608))  # Optional reshaping (commented out)
    ])
    return transform


def define_val_test_transformer():
    """
    Build the image transformation pipeline used for validation and test data.

    No augmentation is applied; only the minimal preprocessing required to
    produce a valid model input:
    - Grayscale conversion (1 channel).
    - Conversion to a PyTorch ``FloatTensor`` scaled to [0, 1].

    Returns:
        torchvision.transforms.Compose: Composed transform for validation / test images.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to single-channel grayscale
        transforms.ToTensor(),                         # Convert PIL image to FloatTensor in [0, 1]
        # transforms.Lambda(lambda x: x.reshape(1, 4608, 4608))  # Optional reshaping (commented out)
    ])
    return transform


def sampler(dataset):
    """
    Build a ``WeightedRandomSampler`` that balances class representation per epoch.

    Assigns each sample a weight inversely proportional to its class frequency
    so that minority classes are drawn more often, counteracting class imbalance.
    Sampling is performed *with replacement*.

    Args:
        dataset (torch.utils.data.Dataset): Dataset with a ``targets`` attribute
            containing integer class labels for every sample.

    Returns:
        torch.utils.data.sampler.WeightedRandomSampler: Class-balanced sampler.
    """

    # Count how many samples exist per class
    class_counter = dict(Counter(dataset.targets))

    # Compute per-class weights as the inverse of their sample counts
    class_weights = 1 / torch.Tensor([ct for ct in class_counter.values()])

    # Index all samples in the dataset
    dataset_indices = list(range(len(dataset)))

    # Retrieve the integer label for every sample
    targets = torch.Tensor(dataset.targets)[dataset_indices]

    # Map each sample to its class weight
    sample_weights = [class_weights[int(target)] for target in targets]

    # Build the sampler; replacement=True allows oversampling of minority classes
    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(dataset), replacement=True
    )

    return weighted_sampler


def split_data_per_type(image_patches, batch_size=16):
    """
    Split image patches into stratified training, validation, and test sets
    based on cancer status, then create corresponding PyTorch DataLoaders.

    Cancerous and non-cancerous patch groups are shuffled independently and
    divided across the three splits while maintaining their class proportions.
    Training uses a weighted sampler to balance class representation, while
    validation and test sets are loaded without sampling or augmentation.

    Args:
        image_patches (dict):
            Dictionary mapping patch group identifiers to their corresponding
            image patch data.
        batch_size (int, optional):
            Number of images loaded per batch. Defaults to 16.

    Returns:
        tuple:
            A tuple containing the training, validation, and test DataLoaders.
    """

    # Create directories for the full dataset and train/validation/test splits
    data_dir, train_dir, val_dir, test_dir = create_data_directories()

    # Separate patch groups into cancerous and non-cancerous categories
    cancer = {}
    no_cancer = {}

    # Separate image patch groups into cancerous and non-cancerous categories
    for key, value in image_patches.items():

        # Identify cancerous groups while excluding keys containing "NotCancerous"
        is_cancer = "Cancerous" in key and "NotCancerous" not in key

        # Identify non-cancerous groups
        is_no_cancer = "NotCancerous" in key

        # Add each patch group to its corresponding class
        if is_cancer:
            cancer[key] = value
        elif is_no_cancer:
            no_cancer[key] = value

    # Extract patch group identifiers for each class
    cancer_keys = list(cancer.keys())
    no_cancer_keys = list(no_cancer.keys())

    # Randomize the order of patch groups before splitting
    random.shuffle(cancer_keys)
    random.shuffle(no_cancer_keys)

    # Split each class while maintaining the desired class proportions
    train_dirs = cancer_keys[:11] + no_cancer_keys[:7]
    val_dirs = cancer_keys[11:16] + no_cancer_keys[7:10]
    test_dirs = cancer_keys[16:] + no_cancer_keys[10:]

    # Shuffle the groups within each split
    random.shuffle(train_dirs)
    random.shuffle(val_dirs)
    random.shuffle(test_dirs)

    # Copy each split's patch groups into the corresponding directories
    copy_directories_to_directories(train_dir, train_dirs)
    copy_directories_to_directories(val_dir, val_dirs)
    copy_directories_to_directories(test_dir, test_dirs)

    # Apply data augmentation only to the training set
    train_transform = define_train_transformer()
    val_transform = define_val_test_transformer()
    test_transform = define_val_test_transformer()

    # Build ImageFolder datasets from the populated split directories
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Use a weighted sampler to balance class representation during training
    weighted_sampler = sampler(train_dataset)

    # Create DataLoaders; pinned memory improves CPU-to-GPU data transfers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=weighted_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader


def split_data_combined(image_patches, batch_size=16):
    """
    Partition image patch directories into train / val / test splits and return DataLoaders.

    The split is performed at the *directory* (patch-group) level rather than
    at the individual file level, so all patches from the same tissue region
    stay together in a single split. Patch groups are separated by dataset
    type (tumor, Voronoi, synthetic) and cancer status, shuffled, and then
    allocated to splits with a fixed count per group so that tumor and
    Voronoi patches from the same region stay aligned across splits.
    Synthetic patches are included only in the training set.

    Steps:
        1. Create clean on-disk directory trees for each split.
        2. Separate patch-group keys by dataset type (tumor / Voronoi /
           synthetic) and cancer status (Cancerous / NotCancerous).
        3. Shuffle tumor patch-group keys and derive the matching Voronoi
           keys for the same tissue regions.
        4. Allocate a fixed number of groups per class to train / val / test,
           then shuffle each split so groups aren't ordered by type.
        5. Copy allocated directories into the appropriate split folders.
        6. Apply augmentation transforms to training data; plain transforms to
           validation and test data.
        7. Wrap each split in a ``DataLoader``.

    Args:
        image_patches (dict): Mapping of patch-group directory path → list of
            image file paths, as returned by ``get_patch_files``.
        total_image_count (int): Total number of image files across all groups.
        batch_size (int): Samples per mini-batch for all DataLoaders. Default 16.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]:
            train_loader, val_loader, test_loader
    """

    # Create clean on-disk directory trees for each split
    data_dir, train_dir, val_dir, test_dir = create_data_directories()

    # Dictionaries to store image patches by dataset type and cancer status
    tumor_cancer = {}
    tumor_no_cancer = {}

    voronoi_cancer = {}
    voronoi_no_cancer = {}

    synthetic_cancer = {}
    synthetic_no_cancer = {}

    # Loop through all image patches and their associated values
    for key, value in image_patches.items():

        # Identify whether the patch belongs to the cancerous or non-cancerous class
        is_cancer = "Cancerous" in key and "NotCancerous" not in key
        is_no_cancer = "NotCancerous" in key

        # Separate tumor dataset patches by cancer status
        if "tumor" in key.lower():
            if is_cancer:
                tumor_cancer[key] = value
            elif is_no_cancer:
                tumor_no_cancer[key] = value

        # Separate Voronoi dataset patches by cancer status
        elif "voronoi" in key.lower():
            if is_cancer:
                voronoi_cancer[key] = value
            elif is_no_cancer:
                voronoi_no_cancer[key] = value

        # Separate synthetic dataset patches by cancer status
        # "context" in the path identifies the synthetic dataset
        elif "context" in key.lower():
            if is_cancer:
                synthetic_cancer[key] = value
            elif is_no_cancer:
                synthetic_no_cancer[key] = value

    # Initialize empty lists for training, validation, and test directories
    train_dirs, val_dirs, test_dirs = [], [], []

    # Get the folder paths for cancerous and non-cancerous tumor images
    tumor_cancer_keys = list(tumor_cancer.keys())
    tumor_no_cancer_keys = list(tumor_no_cancer.keys())

    # Shuffle tumor folders to randomly assign them to train/validation/test sets
    random.shuffle(tumor_cancer_keys)
    random.shuffle(tumor_no_cancer_keys)

    # Build the corresponding Voronoi folder paths for cancerous tumor images
    voronoi_cancer_keys = []

    for tumor_key in tumor_cancer_keys:
        folder_name = os.path.basename(tumor_key)
        # base_voronoi_path = "/Users/arpitha/Documents/Research_Labs/Lab_Schwartz/Lung-Cancer-Image-Classification-main/voronoi_patches/Cancerous/"
        base_voronoi_path = "/ocean/projects/bio240001p/arpitha/Voronoi_patches/Cancerous"
        voronoi_key = base_voronoi_path + folder_name + "_voronoi_diagram"

        # Only include the Voronoi folder if it exists in the dataset
        if voronoi_key in voronoi_cancer:
            voronoi_cancer_keys.append(voronoi_key)

    # Build the corresponding Voronoi folder paths for non-cancerous tumor images
    voronoi_no_cancer_keys = []

    for tumor_key in tumor_no_cancer_keys:
        folder_name = os.path.basename(tumor_key)
        # base_voronoi_path = "/Users/arpitha/Documents/Research_Labs/Lab_Schwartz/Lung-Cancer-Image-Classification-main/voronoi_patches/NotCancerous/"
        base_voronoi_path = "/ocean/projects/bio240001p/arpitha/Voronoi_patches/NotCancerous"
        voronoi_key = base_voronoi_path + folder_name + "_voronoi_diagram"

        # Only include the Voronoi folder if it exists in the dataset
        if voronoi_key in voronoi_no_cancer:
            voronoi_no_cancer_keys.append(voronoi_key)

    # Get the folder paths for synthetic cancerous and non-cancerous images
    synthetic_cancer_keys = list(synthetic_cancer.keys())
    synthetic_no_cancer_keys = list(synthetic_no_cancer.keys())

    # Split the tumor and Voronoi folders into training, validation, and test sets
    # Synthetic images are included only in the training set
    train_dirs = tumor_cancer_keys[:11] + tumor_no_cancer_keys[:7] + voronoi_cancer_keys[:11] + \
                 voronoi_no_cancer_keys[:7] + synthetic_cancer_keys + synthetic_no_cancer_keys

    val_dirs = tumor_cancer_keys[11:16] + tumor_no_cancer_keys[7:10] + \
               voronoi_cancer_keys[11:16] + voronoi_no_cancer_keys[7:10]

    test_dirs = tumor_cancer_keys[16:] + tumor_no_cancer_keys[10:] + \
                voronoi_cancer_keys[16:] + voronoi_no_cancer_keys[10:]

    # Shuffle each split so items aren't grouped by patch type
    random.shuffle(train_dirs)
    random.shuffle(val_dirs)
    random.shuffle(test_dirs)

    # Copy each split's patch groups into the corresponding on-disk directories
    copy_directories_to_directories(train_dir, train_dirs)
    copy_directories_to_directories(val_dir, val_dirs)
    copy_directories_to_directories(test_dir, test_dirs)

    # Training transform includes data augmentation; val/test transforms do not
    train_transform = define_train_transformer()
    val_transform = define_val_test_transformer()
    test_transform = define_val_test_transformer()

    # Build ImageFolder datasets from the populated on-disk split directories
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Weighted sampler ensures balanced class representation during training
    weighted_sampler = sampler(train_dataset)

    # Create DataLoaders; pinned memory improves CPU-to-GPU data transfers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=weighted_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader


class ModelEmbedding(nn.Module):
    """
    ResNet wrapper that exposes both intermediate embeddings and final class logits.

    The original model's final layer is removed. A new linear embedding layer
    (with ReLU activation) is inserted before the classification head, allowing
    both the embedding vector and the class logits to be returned from a single
    forward pass.

    The ``data_type`` argument controls whether the embedding layer is part of
    the classification path:
    - ``"tumor"`` or ``"context"``: Classification head is applied directly to
      the flattened feature map, bypassing the embedding layer.
    - Any other value: Classification head is applied to the embedding output.

    Attributes:
        data_type (str): Segmentation type controlling the forward-pass routing.
        features (nn.Sequential): All original model layers except the final one,
            used as a convolutional feature extractor.
        linear (nn.Linear): Projects the flattened feature map to a
            ``feature_dim``-dimensional embedding space.
        relu (nn.ReLU): Non-linearity applied after ``linear``.
        finlinear (nn.Linear): Classification head mapping from ``feature_dim``
            to ``num_classes`` logits.
    """

    def __init__(self, original_model, data_type):
        """
        Initialise ModelEmbedding from a pre-trained model.

        Args:
            original_model (nn.Module): Pre-trained model whose last child layer
                will be replaced by the custom embedding and classification heads.
            data_type (str): One of ``"tumor"``, ``"context"``, or ``"voronoi"``.
                Controls which path is used inside ``forward``.
        """
        super(ModelEmbedding, self).__init__()

        self.data_type = data_type

        # Strip the original model's final layer; keep everything else as the feature extractor
        self.features = nn.Sequential(*list(original_model.children()))[:-1]

        # Linear layer projecting flattened features to a feature_dim-dimensional embedding
        self.linear = nn.Linear(feature_dim, feature_dim)

        # ReLU non-linearity applied to the embedding output
        self.relu = nn.ReLU(inplace=True)

        # Final classification head: feature_dim → num_classes logits
        self.finlinear = nn.Linear(in_features=feature_dim, out_features=num_classes, bias=True)

    def forward(self, x):
        """
        Run a forward pass and return both the embedding and the class logits.

        Args:
            x (torch.Tensor): Input batch of shape
                ``(batch_size, 1, feature_dim, feature_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                embedding_out – Embedding vector of shape ``(batch_size, feature_dim)``
                    (always computed but used in classification only for non-tumor/context types).
                out – Classification logits of shape ``(batch_size, num_classes)``.
        """
        # Extract feature maps through the convolutional backbone
        embedding = self.features(x)

        # Flatten the spatial feature maps to a 1-D vector per sample
        embedding = embedding.view(embedding.size(0), -1)

        # Compute the embedding vector via a linear transform followed by ReLU
        embedding_out = self.relu(self.linear(embedding))

        if self.data_type in ["tumor", "synthetic"]:
            # Classify directly from the raw flattened features (no embedding layer in path)
            out = self.finlinear(embedding)
        else:
            # Classify from the embedding output (embedding layer is part of the path)
            out = self.finlinear(embedding_out)

        # out = self.finlinear(embedding_out)

        # Return both the embedding vector and the classification logits
        return embedding_out, out


def load_model(data_type):
    """
    Instantiate and configure a modified ResNet18 for single-channel grayscale input.

    Modifications applied to the stock torchvision ResNet18:
    - ``conv1`` replaced to accept 1-channel input instead of 3.
    - ``avgpool`` replaced with ``AdaptiveAvgPool2d(1, 1)`` to handle
      arbitrary spatial input sizes.
    - ``fc`` replaced to output ``num_classes`` logits.
    - The whole model is wrapped in ``ModelEmbedding`` to expose embeddings.
    - Moved to GPU(s) if available; wrapped in ``DataParallel`` for multi-GPU.

    Args:
        data_type (str): Passed through to ``ModelEmbedding`` to control the
            forward-pass routing (``"tumor"``, ``"context"``, or ``"voronoi"``).

    Returns:
        tuple[torch.device, nn.Module]:
            device – The primary computation device (GPU or CPU).
            model  – The fully configured model placed on ``device``.
    """

    # Load ImageNet-pretrained ResNet18 as the backbone
    model = models.resnet18(pretrained=True)

    # Replace the first conv layer to accept single-channel (grayscale) images
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=3,
        stride=2,
        padding=3,
        bias=False
    )

    # Replace the average-pool with an adaptive version so any spatial size is accepted
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    # Swap the final FC layer to produce the correct number of class logits
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Wrap the modified backbone so embeddings and logits can both be retrieved
    model = ModelEmbedding(model, data_type)

    # Use GPU if available; fall back to CPU otherwise
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Distribute computation across all available GPUs when there are multiple
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Move all model parameters and buffers to the selected device
    model = model.to(device)

    return device, model


def get_activation(name, activation):
    """
    Factory that returns a forward hook for capturing a layer's output tensor.

    The returned hook stores a *detached* copy of the layer's output in the
    shared ``activation`` dictionary under ``name``, so it can be read after
    the forward pass without affecting gradient computation.

    Args:
        name (str): Key used to store the captured output in ``activation``.
        activation (dict): Shared dictionary to write captured tensors into.

    Returns:
        Callable: A hook function compatible with ``Module.register_forward_hook``.
    """

    def hook(model, input, output):
        # Detach from the computation graph and store for external access
        activation[name] = output.detach()

    return hook


def get_params(model, learningRate=5e-5, weight_decay=5e-5, momentum=0.70, factor=0.5, patience=3):
    """
    Configure and return the loss function, optimiser, and learning-rate scheduler.

    Args:
        model (nn.Module): The model whose parameters will be optimised.
        learningRate (float): Initial learning rate for AdamW. Default 1e-4.
        weight_decay (float): L2 regularisation coefficient for AdamW. Default 1e-5.
        momentum (float): Kept for API consistency; not used by AdamW. Default 0.70.
        factor (float): Multiplicative LR reduction factor applied by the scheduler.
            Default 0.5 (halves the LR on each trigger).
        patience (int): Number of epochs with no validation-loss improvement before
            the scheduler reduces the LR. Default 3.

    Returns:
        tuple[nn.CrossEntropyLoss, torch.optim.AdamW, ReduceLROnPlateau]:
            criterion  – Cross-entropy loss for multi-class classification.
            optimizer  – AdamW optimiser.
            scheduler  – ``ReduceLROnPlateau`` scheduler monitoring validation loss.
    """

    # Cross-entropy loss is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # AdamW couples weight decay correctly (decoupled from the gradient update)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate, weight_decay=weight_decay)

    # Reduce LR when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

    return criterion, optimizer, scheduler


def train(model, device, train_loader, val_loader, criterion, optimizer, scheduler,
          num_epochs=20, start_epoch=0, all_train_embeddings=[], all_val_embeddings=[],
          all_train_loss=[], all_val_loss=[], all_train_acc=[], all_val_acc=[],
          all_train_f1 = [], all_val_f1 = []):
    """
    Train the model for a fixed number of epochs with per-epoch validation.

    After every training epoch the model is evaluated on the validation set via
    ``testing()``. The learning-rate scheduler is stepped using the validation
    loss at the end of each epoch.

    Per-epoch metrics collected for training:
        - Average cross-entropy loss.
        - Top-1 accuracy.
        - Macro-averaged precision, recall, and F1 (via scikit-learn).
        - Confusion matrix.

    Args:
        model (nn.Module): Model to train (modified in-place).
        device (torch.device): Computation device.
        train_loader (DataLoader): Iterable over training mini-batches.
        val_loader (DataLoader): Iterable over validation mini-batches.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Parameter update rule.
        scheduler (lr_scheduler or None): LR scheduler stepped on validation loss.
            Pass ``None`` to disable scheduling.
        num_epochs (int): Total number of epochs to run. Default 45.
        start_epoch (int): First epoch index (used when resuming training). Default 0.
        all_train_embeddings (list): Accumulator for per-epoch training embeddings.
        all_val_embeddings (list): Accumulator for per-epoch validation embeddings.
        all_train_loss (list): Accumulator for per-epoch mean training loss.
        all_val_loss (list): Accumulator for per-epoch mean validation loss.
        all_train_acc (list): Accumulator for per-epoch training accuracy.
        all_val_acc (list): Accumulator for per-epoch validation accuracy.

    Returns:
        tuple: Eight lists in order —
            all_train_embeddings, all_val_embeddings,
            all_train_loss, all_val_loss,
            all_train_acc,  all_val_acc,
            batch_latencies, batch_throughputs
    """

    # Lists to record per-batch timing statistics across all epochs
    batch_latencies = []
    batch_throughputs = []

    # Put model in training mode (activates dropout, batch-norm running stats, etc.)
    model.train()

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):

        # ── Reset per-epoch accumulators ───────────────────────────────────────
        # NOTE (bug fix): these were originally outside the epoch loop, which
        # caused accuracy and total to accumulate across epochs, producing
        # inflated accuracy values after the first epoch.
        total = 0
        accuracy = 0
        epoch_train_loss = []   # Per-sample loss values for this epoch
        train_embeddings = []   # Embeddings captured this epoch
        avg_loss = 0.0          # Running sum used for periodic batch-level logging

        # Accumulators for scikit-learn metrics (computed at epoch end)
        all_labels = []
        all_preds  = []

        # Confusion matrix reset each epoch; shape: [num_classes × num_classes]
        train_confusion_matrix = torch.zeros(num_classes, num_classes)

        # ── Batch loop ─────────────────────────────────────────────────────────
        for batch_num, (feats, labels) in enumerate(train_loader):
            # Transfer data to the computation device
            feats, labels = feats.to(device), labels.to(device)

            # Reshape to (batch, 1 channel, H, W) expected by the model
            feats = feats.reshape(-1, 1, feature_dim, feature_dim)

            # Start time
            torch.cuda.synchronize()
            batch_start = time.time()

            # Clear gradients from the previous iteration
            optimizer.zero_grad()

            # Forward pass; model returns (embedding, logits)
            _, outputs = model(feats)

            # Convert logits to predicted class indices via softmax then argmax
            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)  # Ensure 1-D tensor

            # Accumulate confusion matrix counts for this batch
            for t, p in zip(labels.view(-1), pred_labels):
                train_confusion_matrix[t.long(), p.long()] += 1

            # Collect CPU copies for scikit-learn metrics at epoch end
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred_labels.cpu().numpy())

            # Store layer activations captured by the forward hook (if registered)
            if 'avg_pool' in activation:
                train_embeddings.append(activation['avg_pool'].detach().cpu().numpy())

            # Compute cross-entropy loss for this batch
            loss = criterion(outputs, labels.long())

            # Backpropagate gradients through the computation graph
            loss.backward()

            # Apply the computed gradients to update model parameters
            optimizer.step()

            # Record batch wall-clock time and throughput
            torch.cuda.synchronize()
            batch_time = time.time() - batch_start
            batch_latencies.append(batch_time * 1000)
            batch_throughputs.append(feats.size(0) / batch_time)

            # Count correct predictions in this batch
            curr_accuracy = torch.sum(torch.eq(pred_labels, labels)).item()

            # Accumulate epoch-level accuracy and sample count
            accuracy += curr_accuracy
            total += len(labels)

            # Store one loss entry per sample so the epoch mean is sample-weighted
            epoch_train_loss.extend([loss.item()] * feats.size(0))

            # Accumulate for the periodic batch-level log message
            avg_loss += loss.item()

            # Log the average loss every 8 batches
            if (batch_num + 1) % 8 == 0:
                print(f'Training Epoch: {epoch + 1}\tBatch: {batch_num + 1}\tAvg-Loss: {avg_loss / 8:.4f}')
                avg_loss = 0.0  # Reset after logging

            # Explicitly delete tensors that are no longer needed this iteration
            del feats, labels, loss, outputs

        # Release cached GPU memory to avoid fragmentation
        torch.cuda.empty_cache()

        # ── Epoch-level training metrics ───────────────────────────────────────
        avg_train_loss = np.mean(epoch_train_loss) if len(epoch_train_loss) > 0 else 0.0
        avg_train_acc  = accuracy / total if total > 0 else 0.0

        # Macro-averaged sklearn metrics (treats all classes equally regardless of size)
        train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        train_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # Append this epoch's embeddings and scalar metrics to the cross-epoch lists
        all_train_embeddings.extend(train_embeddings)
        all_train_loss.append(avg_train_loss)
        all_train_acc.append(avg_train_acc)
        all_train_f1.append(train_f1)

        # Log training metrics for this epoch
        print(f'Epoch {epoch + 1} - Avg Training Loss: {avg_train_loss:.4f} | '
              f'Accuracy: {avg_train_acc:.4f} | Precision: {train_precision:.4f} | '
              f'Recall: {train_recall:.4f} | F1: {train_f1:.4f}')

        # Display the confusion matrix and per-class accuracy (diagonal / row sum)
        print("Training Confusion Matrix:\n", train_confusion_matrix)
        print("Training Normalized Confusion Matrix (per-class accuracy):\n",
              train_confusion_matrix.diag() / train_confusion_matrix.sum(1))

        print()

        # ── Validation pass for this epoch ────────────────────────────────────
        val_confusion_matrix, val_embeddings, val_loss, val_acc, val_precision, val_recall, val_f1 = testing(
            model, device, val_loader, criterion
        )

        # Append validation metrics to the cross-epoch lists
        all_val_embeddings.extend(val_embeddings)
        all_val_loss.append(val_loss)
        all_val_acc.append(val_acc)
        all_val_f1.append(val_f1)

        # Log validation metrics for this epoch
        print(f'Epoch {epoch + 1} - Avg Validation Loss: {val_loss:.4f} | '
              f'Accuracy: {val_acc:.4f} | Precision: {val_precision:.4f} | '
              f'Recall: {val_recall:.4f} | F1: {val_f1:.4f}')

        # Display the validation confusion matrix and per-class accuracy
        print("Validation Confusion Matrix:\n", val_confusion_matrix)
        print("Validation Normalized Confusion Matrix (per-class accuracy):\n",
              val_confusion_matrix.diag() / val_confusion_matrix.sum(1))

        print()
        print()

        # Step the LR scheduler using the validation loss (reduces LR on plateau)
        if scheduler is not None:
            scheduler.step(val_loss)

    # Return all collected metrics including timing statistics
    return all_train_embeddings, all_val_embeddings, all_train_loss, all_val_loss, all_train_acc, all_val_acc, \
           all_train_f1, all_val_f1, batch_latencies, batch_throughputs


def testing(model, device, val_loader, criterion):
    """
    Evaluate the model on a dataset without updating model parameters.

    Suitable for both validation (called from ``train``) and final test
    evaluation (called from ``main``). The model is temporarily put into
    ``eval`` mode and restored to ``train`` mode before returning.

    Args:
        model (nn.Module): Model to evaluate.
        device (torch.device): Computation device.
        val_loader (DataLoader): DataLoader for the evaluation dataset.
        criterion (nn.Module): Loss function used to compute batch losses.

    Returns:
        tuple:
            val_confusion_matrix (torch.Tensor): Shape ``[num_classes, num_classes]``.
            val_embeddings (list): Per-batch embedding arrays captured via hook.
            val_loss (float): Mean cross-entropy loss over the full dataset.
            val_acc (float): Top-1 accuracy over the full dataset.
            precision (float): Macro-averaged precision (scikit-learn).
            recall (float): Macro-averaged recall (scikit-learn).
            f1 (float): Macro-averaged F1 score (scikit-learn).
    """

    # Disable dropout and use running stats for batch normalisation
    model.eval()

    total = 0            # Total samples evaluated
    accuracy = 0         # Cumulative correct-prediction count
    test_loss = []       # Per-sample loss values
    val_embeddings = []  # Embeddings captured via forward hook

    # Ground-truth and predicted labels collected for scikit-learn metrics
    all_labels = []
    all_preds  = []

    # Confusion matrix accumulator; zeroed once before iterating batches
    val_confusion_matrix = torch.zeros(num_classes, num_classes)

    # Disable gradient tracking for efficiency during evaluation
    with torch.no_grad():
        for batch_num, (feats, labels) in enumerate(val_loader):
            # Reshape input to (batch, 1 channel, H, W)
            feats = feats.reshape(-1, 1, feature_dim, feature_dim)
            feats, labels = feats.to(device), labels.to(device)

            # Forward pass; model returns (embedding, logits)
            _, outputs = model(feats)

            # Predicted class via softmax argmax
            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)

            # Update confusion matrix
            for t, p in zip(labels.view(-1), pred_labels):
                val_confusion_matrix[t.long(), p.long()] += 1

            # Collect CPU copies for scikit-learn metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred_labels.cpu().numpy())

            # Store hook-captured activations if the hook was registered
            if 'avg_pool' in activation:
                val_embeddings.append(activation['avg_pool'].detach().cpu().numpy())

            # Compute batch loss and store one entry per sample
            loss = criterion(outputs, labels.long())
            test_loss.extend([loss.item()] * feats.size()[0])

            # Accumulate accuracy counts
            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            # Free memory by releasing tensors no longer needed
            del feats, outputs, labels, loss, pred_labels

    # Restore model to training mode so it is ready for the next training epoch
    model.train()

    # Compute aggregate metrics over the full evaluation dataset
    val_loss = np.mean(test_loss) if len(test_loss) > 0 else 0.0
    val_acc  = accuracy / total if total > 0 else 0.0

    # Macro-averaged sklearn metrics
    val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return val_confusion_matrix, val_embeddings, val_loss, val_acc, val_precision, val_recall, val_f1


def create_results_directories():
    """
    Create a clean directory tree for saving training results.

    Detects the runtime environment (Docker vs. local) and sets the base path
    accordingly. Any pre-existing results directory is deleted before the new
    structure is created.

    Directory layout:
        <base_path>/
            accuracies/
            losses/
    """

    # Detect Docker environment via the presence of the sentinel file
    in_docker = os.path.exists('/.dockerenv')

    # Choose base results path based on execution environment
    base_path = '/results' if in_docker else '/ocean/projects/bio240001p/arpitha/results'

    # Remove any pre-existing results directory to avoid stale artefacts
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
        print(f"Directory '{base_path}' has been deleted.")
    else:
        print(f"Directory '{base_path}' does not exist.")

    # Recreate the base results directory
    os.makedirs(base_path, exist_ok=True)
    print(f"Directory '{base_path}' created successfully." if os.path.exists(base_path) else f"Failed to create '{base_path}'.")

    # Create subdirectories for accuracy plots and loss plots
    for subdir in ['accuracies', 'f1_scores', 'losses', 'latency', 'throughput', 'memory']:
        path = os.path.join(base_path, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created successfully." if os.path.exists(path) else f"Failed to create '{path}'.")


def plotAccuracy(mode, accuracies):
    """
    Plot accuracy over epochs and save the figure to disk.

    Args:
        mode (str): Descriptive label for the split being plotted
            (e.g. ``"training"`` or ``"validation"``). Used in the plot title,
            legend, and output filename.
        accuracies (list[float]): Accuracy value recorded at each epoch.
    """

    # Graph title and plot title
    graph_title = mode + " Accuracies"
    plot_title  = mode + " Accuracy Over Time"

    # Output path follows the results directory convention
    save_path = f"/ocean/projects/bio240001p/arpitha/results/accuracies/{mode}_accuracy_plot.png"

    # X-axis: epoch indices starting at 1
    time_intervals = list(range(1, len(accuracies) + 1))

    # Create the figure
    plt.figure(figsize=(8, 5))
    plt.plot(time_intervals, accuracies, marker='o', linestyle='-', label=graph_title)
    plt.xlabel("Time Intervals")
    plt.ylabel("Accuracy")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.axis(xmin=0, ymin=0)
    plt.ylim(0, 1)
    plt.savefig(save_path)


def plotF1Score(mode, f1_scores):
    """
    Plot f1 scores over epochs and save the figure to disk.

    Args:
        mode (str): Descriptive label for the split being plotted
            (e.g. ``"training"`` or ``"validation"``). Used in the plot title,
            legend, and output filename.
        f1_scores (list[float]): F1 score value recorded at each epoch.
    """

    # Graph title and plot title
    graph_title = mode + "F1_Scores"
    plot_title  = mode + " F1 Score Over Time"

    # Output path follows the results directory convention
    save_path = f"/ocean/projects/bio240001p/arpitha/results/f1_scores/{mode}_f1_plot.png"

    # X-axis: epoch indices starting at 1
    time_intervals = list(range(1, len(f1_scores) + 1))

    # Create the figure
    plt.figure(figsize=(8, 5))
    plt.plot(time_intervals, f1_scores, marker='o', linestyle='-', label=graph_title)
    plt.xlabel("Time Intervals")
    plt.ylabel("F1 Scores")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.axis(xmin=0, ymin=0)
    plt.ylim(0, 1)
    plt.savefig(save_path)


def plotLoss(mode, losses):
    """
    Plot loss over epochs and save the figure to disk.

    Args:
        mode (str): Descriptive label for the split being plotted
            (e.g. ``"training"`` or ``"validation"``). Used in the plot title,
            legend, and output filename.
        losses (list[float]): Loss value recorded at each epoch.
    """

    # Graph title and plot title
    graph_title = mode + " Losses"
    plot_title  = mode + " Loss Over Time"

    # Output path follows the results directory convention
    save_path = f"/ocean/projects/bio240001p/arpitha/results/losses/{mode}_loss_plot.png"

    # X-axis: epoch indices starting at 1
    time_intervals = list(range(1, len(losses) + 1))

    # Create the figure
    plt.figure(figsize=(8, 5))
    plt.plot(time_intervals, losses, marker='o', linestyle='-', label=graph_title)
    plt.xlabel("Time Intervals")
    plt.ylabel("Loss")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.axis(xmin=0, ymin=0)
    plt.ylim(0, 1)
    plt.savefig(save_path)


def plotLatency(num_gpus, latencies):
    """
    Plot latencies per-batch and save the figure to disk.

    Args:
        num_gpus (int): Number of GPUs used; included in the saved plot filename.
        latencies (list[float]): Latency values to plot.
    """

    # Graph title and plot title
    graph_title = "Latencies"   # Legend label
    plot_title  = "Latency Over Time"  # Plot title

    # Output path follows the results directory convention
    save_path = f"/ocean/projects/bio240001p/arpitha/results/latency/{num_gpus}_latency_plot.png"

    # X-axis: time intervals corresponding to each latency measurement
    time_intervals = list(range(1, len(latencies) + 1))

    # Create the figure
    plt.figure(figsize=(8, 5))
    plt.plot(time_intervals, latencies, marker='o', linestyle='-', label=graph_title)
    plt.xlabel("Batch Intervals")
    plt.ylabel("Latency (s)")  # Specify units
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.axis(xmin=0, ymin=0)
    plt.ylim(0, 200)
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory


def plotThroughput(num_gpus, throughputs):
    """
    Plot throughputs per-batch and save the figure to disk.

    Args:
        num_gpus (int): Number of GPUs used; included in the saved plot filename.
        throughputs (list[float]): Throughput values to plot.
    """

    # Graph title and plot title
    graph_title = "Throughputs"   # Legend label
    plot_title  = "Throughput Over Time"  # Plot title

    # Output path follows the results directory convention
    save_path = f"/ocean/projects/bio240001p/arpitha/results/throughput/{num_gpus}_throughput_plot.png"

    # X-axis: time intervals corresponding to each throughput measurement
    time_intervals = list(range(1, len(throughputs) + 1))

    # Create the figure
    plt.figure(figsize=(8, 5))
    plt.plot(time_intervals, throughputs, marker='o', linestyle='-', label=graph_title)
    plt.xlabel("Batch Intervals")
    plt.ylabel("Throughput (samples/sec)")  # Specify units
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.axis(xmin=0, ymin=0)
    plt.savefig(save_path)
    plt.close()


def plotMemories(num_gpus, peak_mem_per_gpu):
    """
    Plot peak GPU memory usage per GPU and save the figure to disk.

    Args:
        num_gpus (int): Number of GPUs used; included in the saved plot filename.
        peak_mem_per_gpu (list[float]): Peak memory (in MB) for each GPU to plot.
    """

    # Graph title and plot title (can include mode if defined globally)
    graph_title = "Peak GPU Memories"   # Legend label
    plot_title  = "GPU Memory Usage Over Time"  # Plot title

    # Output path follows the results directory convention
    save_path = f"/ocean/projects/bio240001p/arpitha/results/memory/{num_gpus}_gpu_memory_plot.png"

    # X-axis: GPU indices (1 to num_gpus) or time intervals if collected per batch
    gpu_indices = list(range(1, num_gpus + 1))

    # Create the figure
    plt.figure(figsize=(8, 5))
    plt.plot(gpu_indices, peak_mem_per_gpu, marker='o', linestyle='-', label=graph_title)
    plt.xlabel("GPU Index")
    plt.ylabel("Peak Memory (MB)")  # Specify units
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.xticks(gpu_indices)
    plt.savefig(save_path)
    plt.close()


def main():
    """
    Entry point: configure the experiment, run training, evaluate, and save results.

    Workflow:
        1. Parse ``--type`` CLI argument (``"tumor"``, ``"voronoi"``, or ``"context"``)
           to select which set of segmentation patches to load.
        2. Resolve patch directory paths for cancerous and non-cancerous images
           based on the runtime environment (Docker or local HPC).
        3. Collect image file paths and the total file count across both classes.
        4. Partition data into train / val / test splits and build DataLoaders.
        5. Instantiate the modified ResNet18 model and move it to the available device.
        6. Register a forward hook on layer 8 of the feature extractor to capture
           ``avg_pool`` activations as embeddings.
        7. Configure loss, optimiser, and LR scheduler.
        8. Train the model, collecting per-epoch embeddings, losses, and accuracies.
        9. Evaluate the trained model on the held-out test set and print all metrics.
        10. Create results directories and save accuracy / loss plots.
        11. Print the total script runtime and the final test F1 score.
    """

    # Set a random seed for full reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Record start time to measure total end-to-end runtime
    start_time = time.time()

    # ── CLI argument parsing ───────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Segmentation input type")
    parser.add_argument('--type', type=str, choices=["tumor", "voronoi", "synthetic", "combined"], required=True)
    args = parser.parse_args()

    # ── Resolve patch directory paths ─────────────────────────────────────────
    # Determine the location of the image patch directories depending on whether
    # the code is running inside Docker or directly on the HPC filesystem.
    # The variables are stored as lists because the "combined" experiment can use
    # multiple image sources simultaneously (real + Voronoi + synthetic).
    if os.path.exists('/.dockerenv'):
        # Paths inside a Docker container

        if args.type == "tumor":
            # Use only real histopathology images
            cancer_patches_dirs = ['/tumor_patches/Cancerous']
            no_cancer_patches_dirs = ['/tumor_patches/NotCancerous']

        elif args.type == "voronoi":
            # Use only Voronoi-transformed images
            cancer_patches_dirs = ['/voronoi_patches/Cancerous']
            no_cancer_patches_dirs = ['/voronoi_patches/NotCancerous']

        elif args.type == "synthetic":
            # Use only synthetic generated images
            cancer_patches_dirs = ['/context_patches/Cancerous']
            no_cancer_patches_dirs = ['/context_patches/NotCancerous']

        elif args.type == "combined":
            # Use all three image sources:
            # real histopathology + Voronoi images + synthetic images
            cancer_patches_dirs = [
                '/tumor_patches/Cancerous',
                '/voronoi_patches/Cancerous',
                '/context_patches/Cancerous'
            ]

            no_cancer_patches_dirs = [
                '/tumor_patches/NotCancerous',
                '/voronoi_patches/NotCancerous',
                '/context_patches/NotCancerous'
            ]

    else:
        # Paths on the local HPC filesystem

        # Base directory containing all image patch folders
        # base_path = "/Users/arpitha/Documents/Research_Labs/Lab_Schwartz/Lung-Cancer-Image-Classification-main"
        base_path = '/ocean/projects/bio240001p/arpitha'

        if args.type == "tumor":
            # Use only real histopathology images
            cancer_patches_dirs = [
                f'{base_path}/tumor_patches/Cancerous'
            ]
            no_cancer_patches_dirs = [
                f'{base_path}/tumor_patches/NotCancerous'
            ]

        elif args.type == "voronoi":
            # Use only Voronoi-transformed images
            cancer_patches_dirs = [
                f'{base_path}/voronoi_patches/Cancerous'
            ]
            no_cancer_patches_dirs = [
                f'{base_path}/voronoi_patches/NotCancerous'
            ]

        elif args.type == "synthetic":
            # Use only synthetic generated images
            cancer_patches_dirs = [
                f'{base_path}/context_patches/Cancerous'
            ]
            no_cancer_patches_dirs = [
                f'{base_path}/context_patches/NotCancerous'
            ]

        elif args.type == "combined":
            # Use all three image sources:
            # real histopathology + Voronoi images + synthetic images
            cancer_patches_dirs = [
                f'{base_path}/tumor_patches/Cancerous',
                f'{base_path}/voronoi_patches/Cancerous',
                f'{base_path}/context_patches/Cancerous'
            ]

            no_cancer_patches_dirs = [
                f'{base_path}/tumor_patches/NotCancerous',
                f'{base_path}/voronoi_patches/NotCancerous',
                f'{base_path}/context_patches/NotCancerous'
            ]

    # ── Collect image patch metadata ──────────────────────────────────────────

    # Initialize a dictionary to store image patch directories and their associated
    # image file paths. Each key corresponds to a patch group directory, and each
    # value contains the list of image files found within that directory.
    image_patches = {}

    # Track the total number of image patches collected across all image sources
    # (tumor, voronoi, synthetic, or combined) and both classes.
    total_image_count = 0

    # Combine all cancerous and non-cancerous patch directories into a single list.
    # This allows the same collection logic to be applied regardless of whether
    # the experiment uses one image source or multiple combined sources.
    all_patch_dirs = cancer_patches_dirs + no_cancer_patches_dirs

    # Iterate through every patch directory and collect image file metadata.
    # The directory names preserve the class information ("Cancerous" or
    # "NotCancerous"), which is used later during dataset organization.
    for patch_dir in all_patch_dirs:
        image_patches, total_image_count = get_patch_files(
            patch_dir,
            image_patches,
            total_image_count
        )


    # Split all image patch types into combined train/validation/test DataLoaders
    if args.type == "combined":
        train_loader, val_loader, test_loader = split_data_combined(image_patches)

    # Split image patches by individual image type into train/validation/test DataLoaders
    else:
        train_loader, val_loader, test_loader = split_data_per_type(image_patches)

    # ── Model instantiation ───────────────────────────────────────────────────
    device, model = load_model(args.type)

    print(f"Model loaded on device: {device}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    # Log per-GPU details when multiple GPUs are in use
    if torch.cuda.device_count() > 1:
        print("GPU Details:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ── Forward hook for embedding extraction ─────────────────────────────────
    # The hook captures the output of layer index 8 in the feature extractor
    # (the adaptive average-pool layer) and stores it in the global `activation` dict.
    hook_function = get_activation('avg_pool', activation)

    if torch.cuda.device_count() > 1:
        # DataParallel wraps the model; access the underlying module's layer
        model.module.features[8].register_forward_hook(hook_function)
    else:
        model.features[8].register_forward_hook(hook_function)
    #
    # ── Optimiser and scheduler setup ─────────────────────────────────────────
    criterion, optimizer, scheduler = get_params(model)

    # ── Training ──────────────────────────────────────────────────────────────
    all_train_embeddings, all_val_embeddings, all_train_loss, all_val_loss, \
    all_train_acc, all_val_acc, all_train_f1, all_val_f1, batch_latencies, \
    batch_throughputs = train(model, device, train_loader, val_loader, criterion, optimizer, scheduler)

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_confusion_matrix, all_test_embeddings, all_test_loss, all_test_acc, test_precision, \
    test_recall, test_f1 = testing(model, device, test_loader, criterion)

    # Print all test-set metrics
    print(f'Average Testing Loss: {all_test_loss:.4f} | '
          f'Accuracy: {all_test_acc:.4f} | Precision: {test_precision:.4f} | '
          f'Recall: {test_recall:.4f} | F1: {test_f1:.4f}')

    # Print test confusion matrix and per-class (normalised) accuracy
    print("Testing Confusion Matrix:\n", test_confusion_matrix)
    print("Testing Normalized Confusion Matrix (per-class accuracy):\n",
          test_confusion_matrix.diag() / test_confusion_matrix.sum(1))

    # ── Log / Print Training Metrics ───────────────────────────────────────────
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()

    # Record the peak GPU memory used on each GPU during this run (in MB)
    peak_mem_per_gpu = [torch.cuda.max_memory_allocated(i) / 1024**2 for i in range(num_gpus)]

    # # Print training metrics
    # # - batch_latencies: list of per-batch latencies
    # # - batch_throughputs: list of per-batch throughput
    # # - peak_mem_per_gpu: max memory used per GPU during the run
    # print(f"Latency: {batch_latencies}\n"
    #       f"Throughput: {batch_throughputs}\n"
    #       f"Memory Usage Per GPU: {peak_mem_per_gpu}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    create_results_directories()

    # Accuracy plots
    plotAccuracy("training", all_train_acc)      # Plot training accuracy per epoch
    plotAccuracy("validation", all_val_acc)     # Plot validation accuracy per epoch

    # F1 score plots
    plotF1Score("training", all_train_f1)       # Plot training macro F1 per epoch
    plotF1Score("validation", all_val_f1)       # Plot validation macro F1 per epoch

    # Loss plots
    plotLoss("training", all_train_loss)        # Plot training loss per epoch
    plotLoss("validation", all_val_loss)        # Plot validation loss per epoch

    # Timing / Performance plots
    plotLatency(num_gpus, batch_latencies)      # Plot batch latencies over time
    plotThroughput(num_gpus, batch_throughputs) # Plot batch throughput over time

    # GPU memory usage plot
    plotMemories(num_gpus, peak_mem_per_gpu)    # Plot peak memory usage per GPU

    # ── Save GPU Performance Metrics to CSV ──────────────────────────────────────────────────────────
    # Compute the average latency across all batches
    # np.sum(batch_latencies) sums the latencies for each batch
    # np.mean(...) calculates the mean of these sums
    average_latency = np.mean(np.sum(batch_latencies))

    # Compute the average throughput across all batches
    # np.sum(batch_throughputs) sums throughput values per batch
    # np.mean(...) calculates the average throughput
    average_throughput = np.mean(np.sum(batch_throughputs))

    # Compute the average peak memory usage per GPU
    # np.sum(peak_mem_per_gpu) sums memory usage per GPU
    # np.mean(...) calculates the average memory usage
    average_memory = np.mean(np.sum(peak_mem_per_gpu))

    # Create a pandas DataFrame to organize the metrics
    # Each metric is placed in a list to make a single-row DataFrame
    df = pd.DataFrame({
         "Number of GPUs": [num_gpus],
         "Average Latencies": [average_latency],
         "Average Throughputs": [average_throughput],
         "Average Memories": [average_memory]
    })

    # Define the folder path
    folder_path = "gpu_metrics"

    # Save the DataFrame to CSV
    csv_path = os.path.join(folder_path, "gpu_metrics.csv")
    df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))

    print(f"CSV saved at: {csv_path}")

    # ── Runtime summary ───────────────────────────────────────────────────────
    end_time = time.time()
    total_time = end_time - start_time
    print("Total Script Runtime: {:.2f} seconds".format(total_time))

    # Final test F1 score (also useful for hyperparameter search scripts that parse stdout)
    print(test_f1)


if __name__ == "__main__":
    main()

# ============================================================
# Single Image Type
# ============================================================
#
# Total: 84 cancerous + 52 non-cancerous = 136 images
#
# Training:   44 cancerous + 28 non-cancerous = 72 images
# Validation: 20 cancerous + 12 non-cancerous = 32 images
# Testing:    20 cancerous + 12 non-cancerous = 32 images


# ============================================================
# Multiple Image Types
# ============================================================
#
# Training:
#   84 synthetic cancer + 52 synthetic no cancer
#   44 tumor cancer + 28 tumor no cancer
#   44 Voronoi cancer + 28 Voronoi no cancer
#   Total: 280 images
#
# Validation:
#   20 tumor cancer + 12 tumor no cancer
#   20 Voronoi cancer + 12 Voronoi no cancer
#   Total: 64 images
#
# Testing:
#   20 tumor cancer + 12 tumor no cancer
#   20 Voronoi cancer + 12 Voronoi no cancer
#   Total: 64 images
#
# Split Percentages:
# Training:   280/408 = 68.63%
# Validation:  64/408 = 15.69%
# Testing:     64/408 = 15.69%
#
# Class Distribution:
# Training:   172 cancer (61.43%) + 108 no cancer (38.57%)
# Validation: 40 cancer (62.50%) + 24 no cancer (37.50%)
# Testing:    40 cancer (62.50%) + 24 no cancer (37.50%)
