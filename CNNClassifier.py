# Import packages
import os
import torch
import random
import shutil
import tifffile
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from torch import optim
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
feature_dim = 200

# Define the number of output classes (e.g., drug labels)
num_classes = 2

# Dictionary to store activations (output features) of specific layers during the forward pass
activation = {}

def get_patch_files(patches_dir):
    # Initialize a dictionary to store image patches categorized by their directories
    image_patches = {}

    # Initialize a counter to keep track of the total number of image files found
    total_image_count = 0

    # Define the suffixes for cancerous and non-cancerous directories
    cancerous_suffix = "/Cancerous"
    not_cancerous_suffix = "/NotCancerous"

    # Iterate through the names of directories in the specified patches directory
    for dir_name in os.listdir(patches_dir):
        # Construct full paths for cancerous and non-cancerous directories
        cancer_dir_name = dir_name + cancerous_suffix
        notCancer_dir_name = dir_name + not_cancerous_suffix

        dir_path = os.path.join(patches_dir, dir_name)
        cancer_dir_path = os.path.join(patches_dir, cancer_dir_name)
        notCancer_dir_path = os.path.join(patches_dir, notCancer_dir_name)

        # Check if the cancerous directory exists and list its files
        if os.path.isdir(cancer_dir_path):
            cancer_image_files = ['Cancerous/' + file for file in os.listdir(cancer_dir_path)]
        else:
            cancer_image_files = []  # Initialize empty list if directory doesn't exist

        # Check if the non-cancerous directory exists and list its files
        if os.path.isdir(notCancer_dir_path):
            notCancer_image_files = ['NotCancerous/' + file for file in os.listdir(notCancer_dir_path)]
        else:
            notCancer_image_files = []  # Initialize empty list if directory doesn't exist

        # Combine image files based on the presence of cancerous and non-cancerous images
        if len(cancer_image_files) > 0 and len(notCancer_image_files) == 0:
            all_image_files = cancer_image_files  # Only cancerous files available
        elif len(cancer_image_files) == 0 and len(notCancer_image_files) > 0:
            all_image_files = notCancer_image_files  # Only non-cancerous files available
        elif len(cancer_image_files) > 0 and len(notCancer_image_files) > 0:
            all_image_files = cancer_image_files + notCancer_image_files  # Both types of files
        else:
            all_image_files = []  # No files found

        # Check if the directory is valid and store the image files in the dictionary
        if os.path.isdir(dir_path):
            image_patches[dir_path] = all_image_files  # Map directory path to its image files
            total_image_count += len(all_image_files)  # Increment the total image count

    # Return the dictionary of image patches and the total count of images found
    return image_patches, total_image_count

def create_data_directories():
    # Determine the base data directory based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the base data directory for the Docker environment
        data_dir = '/data'
    else:
        # Set the base data directory for local development
        data_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/data'

    # Check if the base data directory exists
    if os.path.exists(data_dir):
        # Remove the directory and all its contents if it exists
        shutil.rmtree(data_dir)
        print(f"Directory '{data_dir}' has been deleted.")
    else:
        # Notify if the directory doesn't exist
        print(f"Directory '{data_dir}' does not exist.")

    # Create the base data directory
    os.makedirs(data_dir, exist_ok=True)

    # Verify the creation of the base data directory and print the status
    if os.path.exists(data_dir):
        print(f"Directory '{data_dir}' created successfully.")
    else:
        # Notify if the directory creation fails
        print(f"Failed to create the directory '{data_dir}'.")

    # Determine the training data directory based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the training directory for the Docker environment
        train_dir = '/data/train'
    else:
        # Set the training directory for local development
        train_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/data/train'

    # Create the training directory
    os.makedirs(train_dir, exist_ok=True)

    # Verify the creation of the training directory and print the status
    if os.path.exists(train_dir):
        print(f"Directory '{train_dir}' created successfully.")
    else:
        # Notify if the training directory creation fails
        print(f"Failed to create the directory '{train_dir}'.")

    # Determine the 'Cancerous' sub-directory for training data based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the cancerous training directory for the Docker environment
        cancer_train_dir = '/data/train/Cancerous'
    else:
        # Set the cancerous training directory for local development
        cancer_train_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/data/train/Cancerous'

    # Create the 'Cancerous' training directory
    os.makedirs(cancer_train_dir, exist_ok=True)

    # Verify the creation of the 'Cancerous' training directory and print the status
    if os.path.exists(cancer_train_dir):
        print(f"Directory '{cancer_train_dir}' created successfully.")
    else:
        # Notify if the 'Cancerous' directory creation fails
        print(f"Failed to create the directory '{cancer_train_dir}'.")

    # Determine the 'NotCancerous' sub-directory for training data based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the 'NotCancerous' training directory for the Docker environment
        notCancer_train_dir = '/data/train/NotCancerous'
    else:
        # Set the 'NotCancerous' training directory for local development
        notCancer_train_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/data/train/NotCancerous'

    # Create the 'NotCancerous' training directory
    os.makedirs(notCancer_train_dir, exist_ok=True)

    # Verify the creation of the 'NotCancerous' training directory and print the status
    if os.path.exists(notCancer_train_dir):
        print(f"Directory '{notCancer_train_dir}' created successfully.")
    else:
        # Notify if the 'NotCancerous' directory creation fails
        print(f"Failed to create the directory '{notCancer_train_dir}'.")

    # Determine the validation data directory based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the validation directory for the Docker environment
        val_dir = '/data/val'
    else:
        # Set the validation directory for local development
        val_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/data/val'

    # Create the validation directory
    os.makedirs(val_dir, exist_ok=True)

    # Verify the creation of the validation directory and print the status
    if os.path.exists(val_dir):
        print(f"Directory '{val_dir}' created successfully.")
    else:
        # Notify if the validation directory creation fails
        print(f"Failed to create the directory '{val_dir}'.")

    # Determine the 'Cancerous' sub-directory for validation data based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the cancerous validation directory for the Docker environment
        cancer_val_dir = '/data/val/Cancerous'
    else:
        # Set the cancerous validation directory for local development
        cancer_val_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/data/val/Cancerous'

    # Create the 'Cancerous' validation directory
    os.makedirs(cancer_val_dir, exist_ok=True)

    # Verify the creation of the 'Cancerous' validation directory and print the status
    if os.path.exists(cancer_val_dir):
        print(f"Directory '{cancer_val_dir}' created successfully.")
    else:
        # Notify if the 'Cancerous' validation directory creation fails
        print(f"Failed to create the directory '{cancer_val_dir}'.")

    # Determine the 'NotCancerous' sub-directory for validation data based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the 'NotCancerous' validation directory for the Docker environment
        notCancer_val_dir = '/data/val/NotCancerous'
    else:
        # Set the 'NotCancerous' validation directory for local development
        notCancer_val_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/data/val/NotCancerous'

    # Create the 'NotCancerous' validation directory
    os.makedirs(notCancer_val_dir, exist_ok=True)

    # Verify the creation of the 'NotCancerous' validation directory and print the status
    if os.path.exists(notCancer_val_dir):
        print(f"Directory '{notCancer_val_dir}' created successfully.")
    else:
        # Notify if the 'NotCancerous' validation directory creation fails
        print(f"Failed to create the directory '{notCancer_val_dir}'.")

    # Determine the testing data directory based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the testing directory for the Docker environment
        test_dir = '/data/test'
    else:
        # Set the testing directory for local development
        test_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/data/test'

    # Create the testing directory
    os.makedirs(test_dir, exist_ok=True)

    # Verify the creation of the testing directory and print the status
    if os.path.exists(test_dir):
        print(f"Directory '{test_dir}' created successfully.")
    else:
        # Notify if the testing directory creation fails
        print(f"Failed to create the directory '{test_dir}'.")

    # Determine the 'Cancerous' sub-directory for testing data based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the cancerous testing directory for the Docker environment
        cancer_test_dir = '/data/test/Cancerous'
    else:
        # Set the cancerous testing directory for local development
        cancer_test_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/data/test/Cancerous'

    # Create the 'Cancerous' testing directory
    os.makedirs(cancer_test_dir, exist_ok=True)

    # Verify the creation of the 'Cancerous' testing directory and print the status
    if os.path.exists(cancer_test_dir):
        print(f"Directory '{cancer_test_dir}' created successfully.")
    else:
        # Notify if the 'Cancerous' testing directory creation fails
        print(f"Failed to create the directory '{cancer_test_dir}'.")

    # Determine the 'NotCancerous' sub-directory for testing data based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the 'NotCancerous' testing directory for the Docker environment
        notCancer_test_dir = '/data/test/NotCancerous'
    else:
        # Set the 'NotCancerous' testing directory for local development
        notCancer_test_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/data/test/NotCancerous'

    # Create the 'NotCancerous' testing directory
    os.makedirs(notCancer_test_dir, exist_ok=True)

    # Verify the creation of the 'NotCancerous' testing directory and print the status
    if os.path.exists(notCancer_test_dir):
        print(f"Directory '{notCancer_test_dir}' created successfully.")
    else:
        # Notify if the 'NotCancerous' testing directory creation fails
        print(f"Failed to create the directory '{notCancer_test_dir}'.")

    # Return the paths of the created data directories
    return data_dir, train_dir, val_dir, test_dir

def copy_files_to_directories(dst_dir, dirs):

    # Iterate through each source directory in the provided list of directories
    for dir in dirs:
        # Define the 'Cancerous' source and destination directories
        cancer_dir = dir + "/Cancerous"
        cancer_dst_dir = dst_dir + "/Cancerous"

        # Ensure the destination directory exists
        os.makedirs(cancer_dst_dir, exist_ok=True)

        # Copy all files from the 'Cancerous' subdirectory to the destination
        if os.path.exists(cancer_dir):
            for filename in os.listdir(cancer_dir):
                file_path = os.path.join(cancer_dir, filename)
                if os.path.isfile(file_path):
                    # Copy each file to the destination (print optional)
                    shutil.copy(file_path, cancer_dst_dir)
                    # print(f"Copied {filename} to {cancer_dst_dir}")
        else:
            # Notify if the 'Cancerous' source directory does not exist
            print(f"Source directory '{cancer_dir}' does not exist.")

        # Define the 'NotCancerous' source and destination directories
        notCancer_dir = dir + "/NotCancerous"
        notCancer_dst_dir = dst_dir + "/NotCancerous"

        # Ensure the destination directory exists
        os.makedirs(notCancer_dst_dir, exist_ok=True)

        # Copy all files from the 'NotCancerous' subdirectory to the destination
        if os.path.exists(notCancer_dir):
            for filename in os.listdir(notCancer_dir):
                file_path = os.path.join(notCancer_dir, filename)
                if os.path.isfile(file_path):
                    # Copy each file to the destination (print optional)
                    shutil.copy(file_path, notCancer_dst_dir)
                    # print(f"Copied {filename} to {notCancer_dst_dir}")
        else:
            # Notify if the 'NotCancerous' source directory does not exist
            print(f"Source directory '{notCancer_dir}' does not exist.")

def define_train_transformer():

    # Create a composed transform consisting of multiple transformations
    transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Convert the image to grayscale with 1 output channel (removes color information)
                transforms.RandomHorizontalFlip(),            # Randomly flip the image horizontally (with 50% probability)
                transforms.RandomVerticalFlip(),              # Randomly flip the image vertically (with 50% probability)
                transforms.ColorJitter(                       # Apply random changes to brightness, contrast, saturation, and hue
                brightness=1.5,                           # Increase brightness by a factor of up to 1.5
                contrast=1.0,                             # Keep contrast unchanged
                saturation=1.0,                           # Keep saturation unchanged (since it's grayscale, this won't have much effect)
                hue=0                                     # Shift hue by up to 1.0 (maximum hue shift)
                ),
                transforms.ToTensor(),                        # Convert the processed image to a PyTorch tensor for model input
                # transforms.Lambda(lambda x: x.reshape(1, 4608, 4608))  # (Optional) Reshape if needed, comment left in case reshaping is required
    ])

    # Return the defined transformation
    return transform

def define_val_test_transformer():

    # Create a composed transform for validation and testing dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert the image to grayscale with 1 output channel (removing color information)
        transforms.ToTensor(),                          # Convert the processed image to a PyTorch tensor for model input
        # transforms.Lambda(lambda x: x.reshape(1, 4608, 4608))  # (Optional) Reshape if needed; left commented for potential future use
    ])

    # Return the defined transformation for testing
    return transform

def sampler(dataset, count, size=1000):

    # Count the occurrences of each class in the dataset using Counter
    class_counter = dict(Counter(dataset.targets))

    # Calculate class weights as the inverse of the class frequencies.
    # This gives higher weight to underrepresented classes,
    # ensuring they are sampled more frequently.
    class_weights = 1 / torch.Tensor([ct for ct in class_counter.values()])

    # Create a list of indices for the training dataset.
    # Use the count parameter to generate indices from 0 to count-1.
    dataset_indices = list(range(count))

    # Retrieve the targets (labels) for the training dataset using the calculated indices.
    targets = torch.Tensor(dataset.targets)[dataset_indices]

    # Create a WeightedRandomSampler to sample from the dataset.
    # The sampler uses the calculated sample weights to ensure balanced sampling.
    # The `size` parameter determines how many samples to draw from the dataset.
    # `replacement=True` allows the same sample to be drawn multiple times,
    # enabling the selection of underrepresented classes more frequently.
    sample_weights = []
    for target in targets:
        sample_weights.append(class_weights[int(target)])

    # Initialize the WeightedRandomSampler with the calculated sample weights
    # and the specified size for the number of samples to draw.
    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, size, replacement=True)

    # Return the weighted sampler
    return weighted_sampler

def splitting_data(image_patches, total_image_count, train_ratio=0.75, val_ratio=0.17, test_ratio=0.13, batch_size=32):

    # Create directories for training, validation, and testing data
    data_dir, train_dir, val_dir, test_dir = create_data_directories()

    # Get the keys from the image patches and shuffle them
    image_keys = list(image_patches.keys())
    random.shuffle(image_keys)  # Randomly shuffle keys to ensure a diverse split

    # Sort the image patches based on shuffled keys
    sorted_image_patches = {key: image_patches[key] for key in image_keys}

    # Calculate the number of images for training, validation, and testing
    train_count = round(train_ratio * total_image_count)
    val_count = round(val_ratio * total_image_count)
    test_count = round(test_ratio * total_image_count)

    # Initialize lists to store directories for each dataset
    train_dirs = []
    val_dirs = []
    test_dirs = []

    # Initialize counters for current counts of images in each dataset
    cur_train_count = 0
    cur_val_count = 0
    cur_test_count = 0

    # Distribute image files into training, validation, and testing directories
    for image_path, image_files in sorted_image_patches.items():
        count = len(image_files)  # Get the number of image files in the current directory
        # Check if we can add to the training dataset
        if (cur_train_count + count) < train_count:
            cur_train_count += count  # Update current training count
            train_dirs.append(image_path)  # Add current path to training directories
        else:
            # If not enough space in training, check validation dataset
            if (cur_val_count + count) < val_count:
                cur_val_count += count  # Update current validation count
                val_dirs.append(image_path)  # Add current path to validation directories
            else:
                # Otherwise, add to testing dataset
                cur_test_count += count  # Update current testing count
                test_dirs.append(image_path)  # Add current path to testing directories

    # Copy the image files to their respective locations
    copy_files_to_directories(train_dir, train_dirs)
    copy_files_to_directories(val_dir, val_dirs)
    copy_files_to_directories(test_dir, test_dirs)

    # Define the transformations to be applied to the training, validation, and testing images
    train_transform = define_train_transformer()  # Transformations for the training dataset
    val_transform = define_val_test_transformer()  # Transformations for the validation dataset
    test_transform = define_val_test_transformer() # Transformations for the testing dataset

    # Create datasets for training, validation, and testing

    # Load the training dataset using ImageFolder and apply the specified transformations.
    # Create a weighted sampler for the training dataset to ensure balanced class representation.
    # Initialize a DataLoader for the training dataset with the specified batch size.
    # The weighted sampler is used to sample data points, allowing for more frequent sampling of underrepresented classes.
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    weighted_sampler = sampler(train_dataset, cur_train_count)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_sampler)

    # Load the validation dataset using ImageFolder and apply the specified transformations.
    # Initialize a DataLoader for the validation dataset with the specified batch size.
    # The validation DataLoader does not use a sampler, ensuring the data is drawn sequentially.
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Load the testing dataset using ImageFolder and apply the specified transformations.
    # Initialize a DataLoader for the testing dataset with the specified batch size.
    # Similar to the validation loader, this does not use a sampler for sequential access.
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Return the DataLoader objects for training, validation, and testing datasets.
    return train_loader, val_loader, test_loader

class CustomCNN(nn.Module):
    """
    A custom Convolutional Neural Network (CNN) designed for classification tasks.
    This model consists of four convolutional layers followed by two fully connected layers.

    Args:
        num_classes (int): The number of output classes for classification.
    """

    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        # First convolutional layer:
        # Input channels = 1 (grayscale images), Output channels = 16,
        # Kernel size = 3x3, Stride = 1, Padding = 1 (to maintain image size)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Second convolutional layer:
        # Input channels = 16 (from conv1), Output channels = 32,
        # Kernel size = 3x3, Stride = 1, Padding = 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Third convolutional layer:
        # Input channels = 32 (from conv2), Output channels = 48,
        # Kernel size = 3x3, Stride = 1, Padding = 1
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Fourth convolutional layer:
        # Input channels = 48 (from conv3), Output channels = 64,
        # Kernel size = 3x3, Stride = 1, Padding = 1
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Max pooling layer with a kernel size of 2x2 and stride 2 to downsample the image
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # The input size for the first fully connected layer (fc1) after pooling.
        # Assuming the input image size is (100x100), the size after four layers of pooling:
        # (100 / 2) / 2 / 2 / 2 = 6.25, which we round to 6 (or use adaptive pooling for exact size).
        # So, fc1_input_size = 64 channels * 6 * 6
        self.fc1_input_size = 64 * 12 * 12

        # Fully connected layer 1:
        # Takes the flattened input from the conv layers and outputs num_classes features
        self.fc1 = nn.Linear(self.fc1_input_size, 128)  # Reduced the size for more manageable output

        # Fully connected layer 2:
        # Final output layer, matching the number of output classes (num_classes)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, image_height, image_width).

        Returns:
            torch.Tensor: The output tensor representing class scores or logits.
        """

        # Pass the input through the first convolutional layer, apply ReLU activation, and pool
        x = self.pool(F.relu(self.conv1(x)))

        # Pass through the second convolutional layer, apply ReLU activation, and pool
        x = self.pool(F.relu(self.conv2(x)))

        # Pass through the third convolutional layer, apply ReLU activation, and pool
        x = self.pool(F.relu(self.conv3(x)))

        # Pass through the fourth convolutional layer, apply ReLU activation, and pool
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the feature maps from 2D to 1D (batch_size, flattened_features) before passing to the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten dynamically based on the batch size

        # Pass through the first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Pass through the second fully connected layer to produce final output (logits)
        x = self.fc2(x)

        return x

def load_model():

    # Initialize the custom CNN model with the number of output classes.
    model = CustomCNN(num_classes)

    # Determine whether to use a GPU (if available) or fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transfer the model to the specified device (either GPU or CPU).
    model.to(device)

    # Return the device and the modified model.
    return device, model

def get_params(model, learningRate=1e-4, weight_decay=5e-5, momentum=0.9, factor=0.7, patience=3):

    # Define the loss function as CrossEntropyLoss, commonly used for multi-class classification tasks.
    criterion = nn.CrossEntropyLoss()

    # Initialize the Adam optimizer for the model's parameters with the specified learning rate.
    # The 'momentum' argument is not applicable for Adam but is included for consistency with other optimizers.
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weight_decay)

    # Set up a learning rate scheduler that reduces the learning rate when a plateau in validation loss is detected.
    # The 'factor' specifies how much to reduce the learning rate.
    # The 'patience' defines how many epochs to wait before reducing the learning rate if no improvement is seen.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience)

    # Return the defined loss function, optimizer, and learning rate scheduler.
    return criterion, optimizer, scheduler

def train(model, device, train_loader, val_loader, criterion, optimizer, scheduler,
          num_epochs=3, start_epoch=0, all_train_loss=[], all_val_loss=[], all_train_acc=[], all_val_acc=[]):
    """
    Main function for training the model.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        device (torch.device): The device to train the model on (CPU or GPU).
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        scheduler (torch.optim.lr_scheduler): The scheduler for dynamically adjusting the learning rate.
        num_epochs (int): The number of epochs to train the model.
        start_epoch (int, optional): The epoch from which to resume training. Default is 0.
        all_train_loss (list, optional): List to store the training losses for each epoch. Default is an empty list.
        all_val_loss (list, optional): List to store the validation losses for each epoch. Default is an empty list.
        all_train_acc (list, optional): List to store the training accuracies for each epoch. Default is an empty list.
        all_val_acc (list, optional): List to store the validation accuracies for each epoch. Default is an empty list.

    Returns:
        tuple: A tuple containing:
            - all_train_embeddings (list): Extracted embeddings from training data.
            - all_val_embeddings (list): Validation embeddings from validation data.
            - all_train_loss (list): Training losses for each epoch.
            - all_val_loss (list): Validation losses for each epoch.
            - all_train_acc (list): Training accuracies for each epoch.
            - all_val_acc (list): Validation accuracies for each epoch.
    """

    # Set the model to training mode
    model.train()

    # Initialize variables to track total accuracy and loss
    total = 0
    accuracy = 0

    # Initialize lists to store training metrics
    train_loss = []  # To store the training losses

    # Assuming num_classes is defined globally or passed as a parameter
    train_confusion_matrix = torch.zeros(num_classes, num_classes)  # Ensure num_classes is defined

    # Start training over the specified number of epochs
    for epoch in range(start_epoch, num_epochs):

        avg_loss = 0.0  # Initialize average loss for the current epoch

        # Iterate through batches of the training dataset
        for batch_num, (feats, labels) in enumerate(train_loader):

            # Reshape the features to match the input dimensions of the model
            feats = feats.reshape(-1, 1, feature_dim, feature_dim)  # Ensure feature_dim is defined
            feats, labels = feats.to(device), labels.to(device)  # Move to the specified device

            # Clear the gradients for the optimizer
            optimizer.zero_grad()

            # Pass the features through the model
            outputs = model(feats)

            # Get predicted labels by applying softmax and taking the maximum
            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)

            # Update the confusion matrix
            for t, p in zip(labels.view(-1), pred_labels):
                train_confusion_matrix[t.long(), p.long()] += 1

            # Calculate the loss
            loss = criterion(outputs, labels.long())  # Compute the loss
            loss.backward()  # Backpropagate the loss

            # Update the model parameters
            optimizer.step()

            # Store loss for each sample in the batch
            train_loss.extend([loss.item()] * feats.size(0))

            # Calculate average loss for the last 8 batches
            avg_loss += loss.item()
            if (batch_num + 1) % 8 == 0:
                print('Training Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch + 1, batch_num + 1, avg_loss / 8))
                avg_loss = 0.0  # Reset average loss after printing

            # Update accuracy
            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()  # Corrected from pred_labels
            total += len(labels)  # Increment total number of samples

            # Free up unused memory on the GPU
            torch.cuda.empty_cache()

            # Delete variables to free up memory after each batch
            del feats, labels, loss, outputs

        # Print the confusion matrix and normalized confusion matrix for accuracy analysis
        print("Training Confusion Matrix:\n", train_confusion_matrix)
        print("Training Normalized Confusion Matrix (per-class accuracy):\n", train_confusion_matrix.diag() / train_confusion_matrix.sum(1))

        # Calculate average training loss and accuracy for the epoch
        avg_train_loss = np.mean(train_loss) if len(train_loss) > 0 else 0  # Use the mean if train_loss is not empty
        avg_train_acc = accuracy / total if total > 0 else 0  # Avoid division by zero

        # Store training metrics
        all_train_loss.append(avg_train_loss)
        all_train_acc.append(avg_train_acc)

        # Assuming `epoch` is the variable that holds the current epoch number
        print(f'Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}, Average Training Accuracy: {avg_train_acc:.4f}')

        # Validate the model on the validation set
        val_confusion_matrix, val_loss, val_acc = testing(model, device, val_loader, criterion)

        # Store validation metrics
        all_val_loss.append(val_loss)
        all_val_acc.append(val_acc)

        # Print the confusion matrix and normalized confusion matrix for accuracy analysis
        print("Validation Confusion Matrix:\n", val_confusion_matrix)
        print("Validation Normalized Confusion Matrix (per-class accuracy):\n", val_confusion_matrix.diag() / val_confusion_matrix.sum(1))

        # Assuming `epoch` is the variable that holds the current epoch number
        print(f'Epoch {epoch + 1} - Average Validation Loss: {val_loss:.4f}, Average Validation Accuracy: {val_acc:.4f}')

        # Print Space
        print()

        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_acc)

    return all_train_loss, all_val_loss, all_train_acc, all_val_acc

def testing(model, device, val_loader, criterion):
    '''
    Calculates the accuracy and loss on the validation dataset.

    Args:
        - model: PyTorch model object, the model being evaluated.
        - val_loader: PyTorch DataLoader object, contains the validation dataset.
        - criterion: PyTorch criterion object, the loss function.

    Returns:
        - val_embeddings: list, the embeddings from the validation dataset.
        - val_loss: float, the average loss on the validation dataset.
        - val_acc: float, the average accuracy on the validation dataset.
    '''

    # Set the model to evaluation mode (disables dropout, batch normalization, etc.)
    model.eval()

    total = 0           # Total number of samples processed
    accuracy = 0        # Total correct predictions accumulator
    test_loss = []      # List to store the loss for each batch/sample

    # Initialize an empty confusion matrix to track model predictions vs true labels
    val_confusion_matrix = torch.zeros(num_classes, num_classes)

    # Iterate over validation data in batches
    for batch_num, (feats, labels) in enumerate(val_loader):

        # Reshape features into the required input shape (batch_size, channels, height, width)
        feats = feats.reshape(-1, 1, feature_dim, feature_dim)  # Reshape the input features to match the model's expected input dimensions
        feats, labels = feats.to(device), labels.to(device)  # Move the features and labels to the specified device (CPU or GPU)

        # Forward pass: Get the model's predictions (outputs)
        outputs = model(feats)

        # Get the predicted labels by finding the index of the maximum output value
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)  # Flatten the predicted labels tensor

        # Update the confusion matrix by comparing true labels with predicted labels
        for t, p in zip(labels.view(-1), pred_labels):
            val_confusion_matrix[t.long(), p.long()] += 1

        # Calculate the loss using the criterion (e.g., cross-entropy loss)
        loss = criterion(outputs, labels.long())

        # Extend the test_loss list by repeating the loss value for each sample in the batch
        test_loss.extend([loss.item()] * feats.size()[0])

        # Add up the number of correct predictions
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()

        # Update the total number of samples processed
        total += len(labels)

        # Free memory by deleting tensors that are no longer needed
        del feats, outputs, labels, loss, pred_labels

    # Switch the model back to training mode after evaluation
    model.train()

    # Calculate the average validation loss across all batches
    val_loss = np.mean(test_loss) if len(test_loss) > 0 else 0 # Use the mean if test_loss is not empty

    # Calculate the overall accuracy on the validation dataset
    val_acc = accuracy / total if total > 0 else 0

    # Return the val embeddings, val loss, and val accuracy
    return val_confusion_matrix, val_loss, val_acc

def main():

    # Determine the patches directory based on the execution environment
    # Check if the code is running inside a Docker environment
    if os.path.exists('/.dockerenv'):
        patches_dir = '/patches'  # Set the patches directory for Docker
    else:
        patches_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/patches'  # Set the patches directory for local development

    # Retrieve the image patches and total image count from the specified directory
    image_patches, total_image_count = get_patch_files(patches_dir)

    # Split the image patches into training, validation, and testing datasets according to specified ratios
    train_loader, val_loader, test_loader = splitting_data(image_patches, total_image_count)

    # Load the model architecture or pre-trained model.
    device, model = load_model()

    # Call the get_params function to obtain the loss function (criterion), optimizer, and learning rate scheduler for the model.
    criterion, optimizer, scheduler = get_params(model)

    # Train the model and retrieve the embeddings, training losses, and training accuracies
    all_train_loss, all_val_loss, all_train_acc, all_val_acc = train(model, device, train_loader, val_loader, criterion, optimizer, scheduler)

    # Run the testing phase of the model on the test dataset
    test_confusion_matrix, all_test_loss, all_test_acc = testing(model, device, test_loader, criterion)

    # Print the confusion matrix for the testing phase
    print("Testing Confusion Matrix:\n", test_confusion_matrix)

    # Print the normalized confusion matrix to evaluate per-class accuracy
    # Normalization is done by dividing the diagonal by the sum of each row
    print("Testing Normalized Confusion Matrix (per-class accuracy):\n",
          test_confusion_matrix.diag() / test_confusion_matrix.sum(1))

    # Print the average testing loss and accuracy metrics
    print(f'Average Testing Loss: {all_test_loss:.4f}, Average Testing Accuracy: {all_test_acc:.4f}')

# Entry point for the script
if __name__ == "__main__":
    main()

# Doubts
  # 1) What size to do images because the layer size of resnet34 is 512? Images are blurry with size 512.
  # 2) How to do the labelling? What did drug response labels mean? Can I label if it is cancerous or not?
  # 3) If I have to look for more data what is recommended: Kaggle, NBIA Data Retriever, Pittsburgh Supercomputer?
