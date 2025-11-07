# Import packages
import os
import time
import copy
import torch
import random
import shutil
import tifffile
import argparse
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from torch import optim
from pathlib import Path
from torch.optim import SGD
from torch.optim import Adam
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler

# Define the global variables for feature dimension, number of classes, and activation
global feature_dim
global num_classes
global activation

# Define the feature dimension for a single-channel input
feature_dim = 512

# Define the number of output classes (e.g., drug labels)
num_classes = 2

# Dictionary to store activations (output features) of specific layers during the forward pass
activation = {}

def get_patch_files(patches_dir, image_patches, total_image_count):
    """
    Retrieves image patch file paths from subdirectories within a given directory.

    This function:
        - Iterates through all subdirectories within the specified `patches_dir`.
        - Collects image file paths (excluding system files like `.DS_Store`).
        - Updates a dictionary (`image_patches`) that maps each subdirectory to its list of image files.
        - Counts the total number of image files processed.

    Args:
        patches_dir (str): The root directory containing subdirectories of image patches.
        image_patches (dict): A dictionary mapping subdirectory paths to lists of image file paths.
        total_image_count (int): A running count of all image files encountered.

    Returns:
        tuple:
            - image_patches (dict): Updated dictionary mapping subdirectory paths to image file lists.
            - total_image_count (int): Updated count of all image files processed.

    Behavior:
        - Prints progress updates, including the directory being processed and its full path.
        - Skips non-directory files inside `patches_dir`.
        - Ensures only valid image files are included in the dictionary.

    Side Effects:
        - Modifies the `image_patches` dictionary in place.
        - Prints debug messages about the directory structure and processing status.
    """

    # Iterate through each item in the specified patches directory
    for dir_name in os.listdir(patches_dir):

        # Print the current directory name for tracking progress
        print(f"Processing directory: {dir_name}")

        # Construct the full absolute path to the current directory
        dir_path = os.path.join(patches_dir, dir_name)
        print(f"Full path: {dir_path}")

        # Only continue if this path is a directory (skip files)
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
    Creates and manages data directories for training, validation, and testing datasets.

    This function:
        - Detects whether running inside a Docker container.
        - Defines base data directory accordingly (local vs Docker).
        - Deletes and recreates the base data directory if it exists.
        - Creates subdirectories for train, validation, and test datasets.
        - Further organizes these datasets into 'Cancerous' and 'NotCancerous' subdirectories.
        - Prints status messages confirming each directory creation.

    Returns:
        tuple: Paths of the created directories:
            - data_dir (str): Base data directory.
            - train_dir (str): Training data directory.
            - val_dir (str): Validation data directory.
            - test_dir (str): Testing data directory.

    Side Effects:
        - Deletes existing data directory if present.
        - Creates multiple nested directories.
        - Prints creation success/failure messages.
    """

    # Detect environment and set base data directory path accordingly
    if os.path.exists('/.dockerenv'):
        data_dir = '/data'  # Docker environment base directory
    else:
        data_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/data'  # Local environment

    # Delete existing base directory if it exists to start fresh
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print(f"Directory '{data_dir}' has been deleted.")
    else:
        print(f"Directory '{data_dir}' does not exist.")

    # Create base data directory
    os.makedirs(data_dir, exist_ok=True)
    print(f"Directory '{data_dir}' created successfully." if os.path.exists(data_dir) else f"Failed to create '{data_dir}'.")

    # Helper function to create dataset subdirectories with 'Cancerous' and 'NotCancerous' folders
    def create_subdirs(base_path):
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
    Copies a list of source directories into categorized target subdirectories under dst_dir.

    For each directory in `dirs`:
    - If its name contains "NotCancerous", copies it into dst_dir/NotCancerous/
    - If its name contains "Cancerous", copies it into dst_dir/Cancerous/
    - Ignores directories that don't match these categories.

    Args:
        dst_dir (str): Destination root directory.
        dirs (list of str): List of source directory paths to copy.

    Side Effects:
        - Creates directories under dst_dir if needed.
        - Copies directories and their contents using shutil.copytree.
        - Prints a confirmation message per copied directory.
    """

    # Define category subdirectories under destination directory
    cancer_dir_path = os.path.join(dst_dir, 'Cancerous')
    no_cancer_dir_path = os.path.join(dst_dir, 'NotCancerous')

    # Iterate over source directories
    for dir in dirs:
        source_dir = Path(dir)

        # Assign target directory based on category found in source directory's name
        if "NotCancerous" in str(dir):
            target_dir = no_cancer_dir_path
        elif "Cancerous" in str(dir):
            target_dir = cancer_dir_path
        else:
            # Skip directories that don't match expected categories
            continue

        # Compose full target path including source directory name
        target_path = os.path.join(target_dir, os.path.basename(source_dir))

        # Copy the entire source directory to target location
        shutil.copytree(source_dir, target_path)

        # Confirm copy operation
        print(f"Directory '{source_dir}' has been copied into '{target_dir}' as '{target_path}'.")

def define_train_transformer():
    """
    Defines a composed image transformation pipeline for training data preprocessing and augmentation.

    Transformations applied:
    - Convert images to grayscale with a single channel (drops color information).
    - Randomly flip images horizontally and vertically to augment data.
    - Apply random brightness adjustment while keeping contrast and saturation unchanged.
    - Convert images to PyTorch tensors for model input.
    - (Optional) reshape the tensor, currently commented out.

    Returns:
        torchvision.transforms.Compose: Transformation pipeline for training images.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
        transforms.RandomHorizontalFlip(),            # Random horizontal flip (50% chance)
        transforms.RandomVerticalFlip(),              # Random vertical flip (50% chance)
        transforms.ColorJitter(
            brightness=1.5,    # Brightness varies up to ±50%
            contrast=1.0,      # Contrast unchanged
            saturation=1.0,    # Saturation unchanged (no effect on grayscale)
            hue=0              # No hue change
        ),
        transforms.ToTensor(),                        # Convert to tensor for PyTorch
        # transforms.Lambda(lambda x: x.reshape(1, 4608, 4608))  # Optional reshaping (commented)
    ])
    return transform

def define_val_test_transformer():
    """
    Defines a composed image transformation pipeline for validation and testing data preprocessing.

    Transformations applied:
    - Convert images to grayscale with a single channel.
    - Convert images to PyTorch tensors.
    - (Optional) reshape tensor, currently commented out.

    Returns:
        torchvision.transforms.Compose: Transformation pipeline for validation/testing images.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
        transforms.ToTensor(),                          # Convert to tensor for PyTorch
        # transforms.Lambda(lambda x: x.reshape(1, 4608, 4608))  # Optional reshaping (commented)
    ])
    return transform

def sampler(dataset, count, size=1000):
    """
    Creates a weighted sampler to balance class representation during training.

    It computes inverse-frequency weights so that underrepresented classes
    are sampled more frequently, and returns a WeightedRandomSampler.

    Args:
        dataset (torch.utils.data.Dataset): Dataset with a 'targets' attribute listing class labels.
        count (int): Number of dataset samples to consider (typically length of dataset).
        size (int, optional): Number of samples to draw in each epoch (default 1000).

    Returns:
        torch.utils.data.sampler.WeightedRandomSampler: Sampler with class-balanced sampling weights.
    """

    # Count how many samples exist per class
    class_counter = dict(Counter(dataset.targets))

    # Compute weights as inverse of class counts for balancing
    class_weights = 1 / torch.Tensor([ct for ct in class_counter.values()])

    # Prepare indices for the dataset
    dataset_indices = list(range(len(dataset)))

    # Get labels for all samples in dataset_indices
    targets = torch.Tensor(dataset.targets)[dataset_indices]

    # Assign a sampling weight to each sample based on its class weight
    sample_weights = [class_weights[int(target)] for target in targets]

    # Create WeightedRandomSampler with replacement to oversample minority classes
    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, size, replacement=True
    )

    return weighted_sampler

def splitting_data(image_patches, total_image_count, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, batch_size=32):
    """
    Split image patches into training, validation, and test sets, then create DataLoaders.

    Steps:
    1. Create directories for train/val/test datasets.
    2. Shuffle and sort input image patch directories.
    3. Allocate directories to train/val/test sets based on desired ratios.
    4. Copy image files to their respective directories.
    5. Define transformations for each set.
    6. Create PyTorch DataLoaders with appropriate samplers and batching.

    Args:
        image_patches (dict): {directory_path: [list_of_image_files]} mapping.
        total_image_count (int): Total number of image files.
        train_ratio (float): Fraction of data for training (default 0.60).
        val_ratio (float): Fraction of data for validation (default 0.20).
        test_ratio (float): Fraction of data for testing (default 0.20).
        batch_size (int): Number of samples per batch in DataLoader (default 32).

    Returns:
        tuple: (train_loader, val_loader, test_loader) PyTorch DataLoader objects.
    """

    # Create directories to store the split datasets
    data_dir, train_dir, val_dir, test_dir = create_data_directories()

    # Get all keys (directories) and shuffle them for randomness
    image_keys = list(image_patches.keys())
    random.shuffle(image_keys)

    # Map shuffled keys back to patches dict for ordered iteration
    sorted_image_patches = {key: image_patches[key] for key in image_keys}

    # Calculate number of images per split based on ratios
    train_count = round(train_ratio * total_image_count)
    val_count = round(val_ratio * total_image_count)
    test_count = round(test_ratio * total_image_count)

    # Lists to hold directories assigned to each split
    train_dirs, val_dirs, test_dirs = [], [], []

    # Counters to keep track of how many images allocated to each split
    cur_train_count = 0
    cur_val_count = 0
    cur_test_count = 0

    # Allocate directories to train/val/test splits ensuring approximate ratios
    for image_path, image_files in sorted_image_patches.items():
        count = len(image_files)
        if (cur_train_count + count) < train_count:
            cur_train_count += count
            train_dirs.append(image_path)
        elif (cur_val_count + count) < val_count:
            cur_val_count += count
            val_dirs.append(image_path)
        else:
            cur_test_count += count
            test_dirs.append(image_path)

    # Create the directory structure again if needed (safe call)
    create_data_directories()

    # Copy image files into their respective directories for each split
    copy_directories_to_directories(train_dir, train_dirs)
    copy_directories_to_directories(val_dir, val_dirs)
    copy_directories_to_directories(test_dir, test_dirs)

    # Define image transforms for training, validation, and testing sets
    train_transform = define_train_transformer()
    val_transform = define_val_test_transformer()
    test_transform = define_val_test_transformer()

    # Load datasets applying transforms
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Create a weighted sampler for training to balance classes
    weighted_sampler = sampler(train_dataset, cur_train_count)

    # Create DataLoaders with batch size and sampler (train) or shuffle (val/test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

class ModelEmbedding(nn.Module):
    '''
    Custom model wrapper that removes the original model's final layer,
    adds a linear embedding layer, and includes a final classification layer.

    This allows simultaneous feature embedding extraction and classification.

    Attributes:
        features (nn.Sequential): Original model layers except the final output layer.
        linear (nn.Linear): Linear layer to transform features into embeddings.
        relu (nn.ReLU): ReLU activation applied after the linear layer.
        finlinear (nn.Linear): Final classification layer producing class logits.
    '''

    def __init__(self, original_model):
        '''
        Initialize ModelEmbedding by modifying the given pre-trained model.

        Args:
            original_model (nn.Module): Pre-trained model to modify by removing final layer.
        '''
        super(ModelEmbedding, self).__init__()

        # Extract all layers except the last from original model to use as feature extractor
        self.features = nn.Sequential(*list(original_model.children()))[:-1]

        # Linear layer to project extracted features to embeddings of size `feature_dim`
        self.linear = nn.Linear(feature_dim, feature_dim)

        # ReLU activation for non-linearity after linear embedding layer
        self.relu = nn.ReLU(inplace=True)

        # Final classification layer outputting logits for `num_classes` classes
        self.finlinear = nn.Linear(in_features=feature_dim, out_features=num_classes, bias=True)

    def forward(self, x):
        '''
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor, typically image batch of shape (batch_size, channels, H, W).

        Returns:
            tuple:
                embedding_out (torch.Tensor): Embedding vector after linear layer and ReLU, shape (batch_size, feature_dim).
                out (torch.Tensor): Classification logits, shape (batch_size, num_classes).
        '''
        # Pass input through feature extractor layers
        embedding = self.features(x)

        # Flatten feature maps to (batch_size, feature_dim)
        embedding = embedding.view(embedding.size(0), -1)

        # Extract embeddings by linear transformation + ReLU activation
        embedding_out = self.relu(self.linear(embedding))

        # Compute classification output from original embedding features
        out = self.finlinear(embedding)

        # Return embeddings and classification output
        return embedding_out, out

def load_model():
    """
    Loads a pre-trained ResNet18 model, modifies it for grayscale image input,
    and prepares it for training or inference with a custom final fully connected layer.

    Steps:
    1. Load ResNet18 pretrained on ImageNet.
    2. Replace first conv layer to accept single-channel grayscale images.
    3. Replace average pooling with adaptive average pooling (output size 1x1).
    4. Modify final fully connected layer to output the desired number of classes.
    5. Wrap model with a custom ModelEmbedding class to enable feature embedding extraction.
    6. Move model to GPU if available, else CPU.

    Returns:
        device (torch.device): The computation device used (GPU or CPU).
        model (torch.nn.Module): The modified ResNet18 model wrapped with ModelEmbedding.
    """

    # Load pretrained ResNet18 model
    model = models.resnet18(pretrained=True)

    # Modify first conv layer for grayscale input (1 channel)
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=3,
        stride=2,
        padding=3,
        bias=False
    )

    # Replace average pooling with adaptive average pooling (output size 1x1)
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    # Modify final fully connected layer for custom number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Wrap model to extract embeddings by removing final layer
    model = ModelEmbedding(model)

    # Select device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transfer model to the chosen device
    model.to(device)

    return device, model

def get_activation(name, activation):
    """
    Creates a forward hook function to capture activations of a given model layer.

    This hook can be registered on any layer to store its output during forward pass.

    Args:
        name (str): Identifier key for the activation dictionary.
        activation (dict): Dictionary to store activations by name.

    Returns:
        function: A hook function that stores the detached output of the layer.
    """

    def hook(model, input, output):
        # Detach output tensor to avoid affecting gradients and store in activation dict
        activation[name] = output.detach()

    return hook

def get_params(model, learningRate=1e-4, weight_decay=1e-4, momentum=0.9, factor=0.5, patience=3):
    """
    Initializes and returns the loss function, optimizer, and learning rate scheduler for model training.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        learningRate (float, optional): Initial learning rate for the optimizer. Default is 1e-4.
        weight_decay (float, optional): Weight decay (L2 regularization) factor. Default is 1e-4.
        momentum (float, optional): Momentum parameter (not used by AdamW, included for consistency). Default is 0.9.
        factor (float, optional): Multiplicative factor to reduce learning rate when scheduler triggers. Default is 0.5.
        patience (int, optional): Number of epochs with no improvement before reducing LR. Default is 3.

    Returns:
        tuple:
            criterion (torch.nn.CrossEntropyLoss): Loss function for classification tasks.
            optimizer (torch.optim.AdamW): AdamW optimizer with specified LR and weight decay.
            scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): LR scheduler that reduces LR on plateau.
    """

    # CrossEntropyLoss is suitable for multi-class classification problems
    criterion = nn.CrossEntropyLoss()

    # AdamW optimizer includes weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate, weight_decay=weight_decay)

    # ReduceLROnPlateau scheduler lowers LR when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

    return criterion, optimizer, scheduler

def train(model, device, train_loader, val_loader, criterion, optimizer, scheduler,
          num_epochs=1, start_epoch=0, all_train_embeddings=[], all_val_embeddings=[],
          all_train_loss=[], all_val_loss=[], all_train_acc=[], all_val_acc=[]):
    """
    Trains the model over multiple epochs, tracking training and validation metrics.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        device (torch.device): Device to run computations on (CPU or GPU).
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer to update model parameters.
        scheduler (torch.optim.lr_scheduler or None): Learning rate scheduler (optional).
        num_epochs (int): Number of epochs to train.
        start_epoch (int, optional): Epoch number to start training from (for resuming).
        all_train_embeddings (list, optional): List to accumulate train embeddings over epochs.
        all_val_embeddings (list, optional): List to accumulate val embeddings over epochs.
        all_train_loss (list, optional): List to accumulate train loss per epoch.
        all_val_loss (list, optional): List to accumulate val loss per epoch.
        all_train_acc (list, optional): List to accumulate train accuracy per epoch.
        all_val_acc (list, optional): List to accumulate val accuracy per epoch.

    Returns:
        tuple: Updated lists containing embeddings, loss, and accuracy metrics for train and val sets.
    """

    # Set model to training mode (enables dropout, batchnorm updates, etc.)
    model.train()

    # Initialize counters for correct predictions and total samples processed
    total = 0
    accuracy = 0

    # Lists to store batch-wise embeddings and losses during training
    train_embeddings = []
    train_loss = []

    # Initialize confusion matrix for training data (num_classes should be globally defined)
    train_confusion_matrix = torch.zeros(num_classes, num_classes)

    # Loop over each epoch
    for epoch in range(start_epoch, num_epochs):
        avg_loss = 0.0  # Running average loss over batches
        losses = []
        accuracies = []

        # Iterate over batches from training DataLoader
        for batch_num, (feats, labels) in enumerate(train_loader):
            # Reshape features to match model input dimensions (batch, channel, height, width)
            feats = feats.reshape(-1, 1, feature_dim, feature_dim)

            # Move features and labels to the specified device (CPU/GPU)
            feats, labels = feats.to(device), labels.to(device)

            # Zero out gradients to avoid accumulation from previous batch
            optimizer.zero_grad()

            # Forward pass through the model: outputs and embeddings (assuming model returns both)
            _, outputs = model(feats)

            # Predict class labels using softmax probabilities
            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)  # Flatten predictions

            # Update confusion matrix counts for this batch
            for t, p in zip(labels.view(-1), pred_labels):
                train_confusion_matrix[t.long(), p.long()] += 1

            # Extract embeddings from model's stored activations (if available)
            if 'avg_pool' in activation:
                train_embeddings.append(activation['avg_pool'].detach().cpu().numpy())

            # Compute loss between model predictions and true labels
            loss = criterion(outputs, labels.long())

            # Backpropagation to compute gradients
            loss.backward()

            # Optimizer step to update model weights
            optimizer.step()

            # Calculate number of correct predictions in this batch
            curr_accuracy = torch.sum(torch.eq(pred_labels, labels)).item()

            # Update running totals
            accuracy += curr_accuracy
            total += len(labels)

            # Record batch loss (repeated for each sample in batch)
            train_loss.extend([loss.item()] * feats.size(0))

            # Accumulate batch loss for averaging
            avg_loss += loss.item()

            # Print average loss every 8 batches for monitoring
            if (batch_num + 1) % 8 == 0:
                print(f'Training Epoch: {epoch + 1}\tBatch: {batch_num + 1}\tAvg-Loss: {avg_loss / 8:.4f}')
                avg_loss = 0.0  # Reset average loss after printing

            # Clear unused GPU memory to prevent leaks
            torch.cuda.empty_cache()

            # Explicitly delete variables to free memory
            del feats, labels, loss, outputs

        # Print confusion matrix and per-class accuracy for training data
        print("Training Confusion Matrix:\n", train_confusion_matrix)
        print("Training Normalized Confusion Matrix (per-class accuracy):\n",
              train_confusion_matrix.diag() / train_confusion_matrix.sum(1))

        # Compute average training loss and accuracy over entire epoch
        avg_train_loss = np.mean(train_loss) if len(train_loss) > 0 else 0
        avg_train_acc = accuracy / total if total > 0 else 0

        # Append current epoch metrics to overall lists
        all_train_embeddings.extend(train_embeddings)
        all_train_loss.append(avg_train_loss)
        all_train_acc.append(avg_train_acc)

        print(f'Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}, Average Training Accuracy: {avg_train_acc:.4f}')

        # Validate the model on validation dataset and collect metrics
        val_confusion_matrix, val_embeddings, val_loss, val_acc = testing(model, device, val_loader, criterion)

        # Append validation metrics to overall lists
        all_val_embeddings.extend(val_embeddings)
        all_val_loss.append(val_loss)
        all_val_acc.append(val_acc)

        # Print validation confusion matrix and per-class accuracy
        print("Validation Confusion Matrix:\n", val_confusion_matrix)
        print("Validation Normalized Confusion Matrix (per-class accuracy):\n",
              val_confusion_matrix.diag() / val_confusion_matrix.sum(1))

        print(f'Epoch {epoch + 1} - Average Validation Loss: {val_loss:.4f}, Average Validation Accuracy: {val_acc:.4f}')
        print()  # Blank line for readability

        # Step learning rate scheduler based on validation accuracy if scheduler provided
        if scheduler is not None:
            scheduler.step(val_acc)

    # Return all collected training and validation metrics
    return all_train_embeddings, all_val_embeddings, all_train_loss, all_val_loss, all_train_acc, all_val_acc

def testing(model, device, val_loader, criterion):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): Device to run computations on (CPU or GPU).
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function to calculate validation loss.

    Returns:
        tuple:
            - val_confusion_matrix (torch.Tensor): Confusion matrix for validation predictions.
            - val_embeddings (list): List of extracted embeddings from validation data.
            - val_loss (float): Average loss over the validation dataset.
            - val_acc (float): Average accuracy over the validation dataset.
    """

    # Set model to evaluation mode (disables dropout, batchnorm updates)
    model.eval()

    total = 0            # Total samples processed
    accuracy = 0         # Count of correct predictions
    test_loss = []       # List to store batch-wise losses
    val_embeddings = []  # List to collect embeddings

    # Initialize confusion matrix of size [num_classes x num_classes]
    val_confusion_matrix = torch.zeros(num_classes, num_classes)

    # Iterate through batches of validation data
    for batch_num, (feats, labels) in enumerate(val_loader):
        # Reshape input to match model's expected input shape
        feats = feats.reshape(-1, 1, feature_dim, feature_dim)
        feats, labels = feats.to(device), labels.to(device)

        # Forward pass to get model outputs
        _, outputs = model(feats)

        # Predict labels by taking the class with max softmax score
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        # Update confusion matrix counts
        for t, p in zip(labels.view(-1), pred_labels):
            val_confusion_matrix[t.long(), p.long()] += 1

        # Extract embeddings if available in activation dict
        if 'avg_pool' in activation:
            val_embeddings.append(activation['avg_pool'].detach().cpu().numpy())

        # Calculate batch loss
        loss = criterion(outputs, labels.long())
        test_loss.extend([loss.item()] * feats.size()[0])  # Append loss for each sample

        # Count correct predictions in batch
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

        # Free memory by deleting unused variables
        del feats, outputs, labels, loss, pred_labels

    # Switch model back to training mode after evaluation
    model.train()

    # Compute average validation loss and accuracy
    val_loss = np.mean(test_loss) if len(test_loss) > 0 else 0
    val_acc = accuracy / total if total > 0 else 0

    return val_confusion_matrix, val_embeddings, val_loss, val_acc

def create_results_directories():
    """
    Creates directories for storing results, accuracies, and losses.

    Behavior:
    - Uses '/results' if running inside Docker (detected by '/.dockerenv'), otherwise uses a local path.
    - Deletes existing results directory before creating a fresh one.
    - Creates subdirectories 'accuracies' and 'losses' inside the results directory.
    - Prints status messages for each operation.

    Returns:
        None
    """

    # Determine if running inside Docker by checking for '/.dockerenv'
    in_docker = os.path.exists('/.dockerenv')

    # Set base results directory accordingly
    base_path = '/results' if in_docker else '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/results'

    # Remove existing results directory if it exists
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
        print(f"Directory '{base_path}' has been deleted.")
    else:
        print(f"Directory '{base_path}' does not exist.")

    # Create the base results directory
    os.makedirs(base_path, exist_ok=True)
    print(f"Directory '{base_path}' created successfully." if os.path.exists(base_path) else f"Failed to create '{base_path}'.")

    # Create accuracies and losses subdirectories
    for subdir in ['accuracies', 'losses']:
        path = os.path.join(base_path, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created successfully." if os.path.exists(path) else f"Failed to create '{path}'.")

def plotAccuracy(mode, accuracies):
    """
    Plot and save the accuracy curve over training or validation time.

    Args:
        mode (str): Label describing the accuracy type (e.g., "Training", "Validation").
        accuracies (list of float): Accuracy values recorded at each time interval (e.g., epoch).

    Behavior:
        - Creates a line plot of accuracy over time.
        - Saves the plot as a PNG file in a designated results directory.
    """

    # Title for legend and plot window based on mode
    graph_title = mode + " Accuracies"
    plot_title = mode + " Accuracy Over Time"

    # Path where the plot image will be saved
    save_path = f"/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/results/accuracies/{mode}_accuracy_plot.png"

    # X-axis values representing time intervals (e.g., epochs)
    time_intervals = list(range(1, len(accuracies) + 1))

    # Set figure size for better visibility
    plt.figure(figsize=(8, 5))

    # Plot accuracy values with markers and line
    plt.plot(time_intervals, accuracies, marker='o', linestyle='-', label=graph_title)

    # Label x-axis (time, epochs, iterations)
    plt.xlabel("Time Intervals")

    # Label y-axis (accuracy values)
    plt.ylabel("Accuracy")

    # Set the title of the plot
    plt.title(plot_title)

    # Show legend to clarify plotted data
    plt.legend()

    # Enable grid for easier visual interpretation
    plt.grid(True)

    # Save the plot to disk
    plt.savefig(save_path)

def plotLoss(mode, losses):
    """
    Plot and save the loss curve over training or validation time.

    Args:
        mode (str): Label describing the loss type (e.g., "Training", "Validation").
        losses (list of float): Loss values recorded at each time interval (e.g., epoch).

    Behavior:
        - Creates a line plot of loss over time.
        - Saves the plot as a PNG file in a designated results directory.
    """

    # Title for legend and plot window based on mode
    graph_title = mode + " Losses"
    plot_title = mode + " Loss Over Time"

    # Path where the plot image will be saved
    save_path = f"/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/results/losses/{mode}_loss_plot.png"

    # X-axis values representing time intervals (e.g., epochs)
    time_intervals = list(range(1, len(losses) + 1))

    # Set figure size for better visibility
    plt.figure(figsize=(8, 5))

    # Plot loss values with markers and line
    plt.plot(time_intervals, losses, marker='o', linestyle='-', label=graph_title)

    # Label x-axis (time, epochs, iterations)
    plt.xlabel("Time Intervals")

    # Label y-axis (loss values)
    plt.ylabel("Loss")

    # Set the title of the plot
    plt.title(plot_title)

    # Show legend to clarify plotted data
    plt.legend()

    # Enable grid for easier visual interpretation
    plt.grid(True)

    # Save the plot to disk
    plt.savefig(save_path)

def main():
    """
    Main entry point of the script.

    Responsibilities:
    - Parses command-line argument '--type' to determine which segmentation patches to process
      (options: 'tumor', 'nuclei', or 'voronoi').
    - Sets paths to cancerous and non-cancerous patch directories based on environment (Docker or local).
    - Loads image patch file paths and counts total images from cancerous and non-cancerous directories.
    - Splits data into training, validation, and test sets and creates DataLoaders.
    - Loads the model and device (CPU/GPU).
    - Sets up a forward hook on a specific model layer to capture embeddings.
    - Obtains loss function, optimizer, and learning rate scheduler.
    - Trains the model, collecting embeddings, losses, and accuracies.
    - Tests the model on the test dataset and prints confusion matrices and performance metrics.
    - Creates directories for saving results.
    - Plots and saves accuracy and loss curves for training and validation.

    Returns:
        tuple: Lists of accuracies for training, validation, and testing datasets.
    """

    # Record the start time of the training run to measure total runtime
    start_time = time.time()

    # Parse input argument to choose the segmentation patch type to use
    parser = argparse.ArgumentParser(description="Segmentation input type")
    parser.add_argument('--type', type=str, choices=["tumor", "nuclei", "voronoi"], required=True)
    args = parser.parse_args()

    # Determine base paths depending on running inside Docker or locally
    if os.path.exists('/.dockerenv'):
        # Docker environment paths for patch directories
        if args.type == "nuclei":
            cancer_patches_dir = '/nuclei_patches/Cancerous'
            no_cancer_patches_dir = '/nuclei_patches/NotCancerous'
        elif args.type == "tumor":
            cancer_patches_dir = '/tumor_patches/Cancerous'
            no_cancer_patches_dir = '/tumor_patches/NotCancerous'
        elif args.type == "voronoi":
            cancer_patches_dir = '/voronoi_patches/Cancerous'
            no_cancer_patches_dir = '/voronoi_patches/NotCancerous'
    else:
        # Local environment paths
        base_path = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist'
        if args.type == "nuclei":
            cancer_patches_dir = f'{base_path}/nuclei_patches/Cancerous'
            no_cancer_patches_dir = f'{base_path}/nuclei_patches/NotCancerous'
        elif args.type == "tumor":
            cancer_patches_dir = f'{base_path}/tumor_patches/Cancerous'
            no_cancer_patches_dir = f'{base_path}/tumor_patches/NotCancerous'
        elif args.type == "voronoi":
            cancer_patches_dir = f'{base_path}/voronoi_patches/Cancerous'
            no_cancer_patches_dir = f'{base_path}/voronoi_patches/NotCancerous'

    # Initialize dictionary and counter to hold image patch paths and total image count
    image_patches = {}
    total_image_count = 0

    # Load image patches from cancerous and non-cancerous directories and update total count
    image_patches, total_image_count = get_patch_files(cancer_patches_dir, image_patches, total_image_count)
    image_patches, total_image_count = get_patch_files(no_cancer_patches_dir, image_patches, total_image_count)

    # Split data into train, val, test sets with DataLoaders
    train_loader, val_loader, test_loader = splitting_data(image_patches, total_image_count)

    # Load model and assign device (CPU or GPU)
    device, model = load_model()

    # Get hook function to capture embeddings from 'avg_pool' layer
    hook_function = get_activation('avg_pool', activation)

    # Register hook on the model's specific layer to capture features during forward pass
    model.features[8].register_forward_hook(hook_function)

    # Setup loss criterion, optimizer, and scheduler for training
    criterion, optimizer, scheduler = get_params(model)

    # Record time immediately before training starts
    mid_time1 = time.time()

    # Calculate the time spent in setup/preprocessing
    setup_time = mid_time1 - start_time

    # Print the setup/preprocessing time in seconds
    print("Time Before Training: {:.2f} seconds".format(setup_time))

    # Train the model and collect embeddings, loss, accuracy metrics
    all_train_embeddings, all_val_embeddings, all_train_loss, all_val_loss, all_train_acc, all_val_acc = train(
        model, device, train_loader, val_loader, criterion, optimizer, scheduler
    )

    # Test the model on the test dataset and collect metrics
    test_confusion_matrix, all_test_embeddings, all_test_loss, all_test_acc = testing(model, device, test_loader, criterion)

    # Record time immediately after training finishes
    mid_time2 = time.time()

    # Calculate the total training time
    training_time = mid_time2 - mid_time1

    # Print the training time in seconds
    print("Training Time: {:.2f} seconds".format(training_time))

    # Print confusion matrix and normalized per-class accuracy
    print("Testing Confusion Matrix:\n", test_confusion_matrix)
    print("Testing Normalized Confusion Matrix (per-class accuracy):\n",
          test_confusion_matrix.diag() / test_confusion_matrix.sum(1))

    # Print average testing loss and accuracy
    print(f'Average Testing Loss: {all_test_loss:.4f}, Average Testing Accuracy: {all_test_acc:.4f}')

    # Create directories to save results, accuracies, and losses
    create_results_directories()

    # Plot and save accuracy curves for training and validation
    plotAccuracy("training", all_train_acc)
    plotAccuracy("validation", all_val_acc)

    # Plot and save loss curves for training and validation
    plotLoss("training", all_train_loss)
    plotLoss("validation", all_val_loss)

    # Record the end time immediately after metrics calculation
    end_time = time.time()

    # Calculate time spent on metrics calculation
    metrics_time = end_time - mid_time2
    print("Time Spent on Metrics Calculation: {:.2f} seconds".format(metrics_time))

    # Calculate total runtime of the entire script
    total_time = end_time - start_time
    print("Total Script Runtime: {:.2f} seconds".format(total_time))

    # Return final test accuracy for further use
    print(all_test_acc)

if __name__ == "__main__":
    main()
