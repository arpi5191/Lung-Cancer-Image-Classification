# Import packages
import os
import copy
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
from pathlib import Path
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

    # Iterate through the subdirectories within the specified patches directory
    for dir_name in os.listdir(patches_dir):

        # Display the name of the current directory being processed
        print(f"Processing directory: {dir_name}")

        # Construct the full path to the current subdirectory
        dir_path = os.path.join(patches_dir, dir_name)
        print(f"Full path: {dir_path}")

        # Only proceed if dir_path is a valid directory
        if not os.path.isdir(dir_path):
            continue

        # List image files in the directory, excluding system files like .DS_Store
        image_files = [
            os.path.join(dir_path, file) for file in os.listdir(dir_path)
            if file != '.DS_Store' and os.path.isfile(os.path.join(dir_path, file))
        ]

        # Update the dictionary with the list of image files for the current directory
        image_patches[dir_path] = image_files

        # Update the total image count
        total_image_count += len(image_files)

    # Return the dictionary of image patches and the total count of images found
    return image_patches, total_image_count

def create_data_directories():
    """
    Creates and manages data directories for training, validation, and testing datasets.

    This function:
        - Detects whether the code is running inside a Docker container.
        - Defines the base data directory accordingly (local or Docker environment).
        - Deletes and recreates the base data directory if it already exists.
        - Creates subdirectories for training, validation, and testing data.
        - Further organizes data into 'Cancerous' and 'NotCancerous' subdirectories within each dataset type.
        - Prints status messages to confirm directory creation.

    Returns:
        tuple: A tuple containing the paths of the created directories:
            - data_dir (str): The base data directory.
            - train_dir (str): Directory for training data.
            - val_dir (str): Directory for validation data.
            - test_dir (str): Directory for testing data.

    Behavior:
        - If the base directory exists, it is removed before being recreated.
        - Uses `os.makedirs` with `exist_ok=True` to ensure directories are created safely.
        - Outputs messages indicating the success or failure of directory creation.

    Side Effects:
        - Removes existing data directory (`data_dir`) if it already exists.
        - Creates multiple subdirectories inside the base data directory.
        - Prints confirmation messages to the console.
    """

    # Determine the base data directory based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the base data directory for the Docker environment
        data_dir = '/voronoi_data'
    else:
        # Set the base data directory for local development
        data_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_data'

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
        train_dir = '/voronoi_data/train'
    else:
        # Set the training directory for local development
        train_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_data/train'

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
        cancer_train_dir = '/voronoi_data/train/Cancerous'
    else:
        # Set the cancerous training directory for local development
        cancer_train_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_data/train/Cancerous'

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
        notCancer_train_dir = '/voronoi_data/train/NotCancerous'
    else:
        # Set the 'NotCancerous' training directory for local development
        notCancer_train_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_data/train/NotCancerous'

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
        val_dir = '/voronoi_data/val'
    else:
        # Set the validation directory for local development
        val_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_data/val'

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
        cancer_val_dir = '/voronoi_data/val/Cancerous'
    else:
        # Set the cancerous validation directory for local development
        cancer_val_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_data/val/Cancerous'

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
        notCancer_val_dir = '/voronoi_data/val/NotCancerous'
    else:
        # Set the 'NotCancerous' validation directory for local development
        notCancer_val_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_data/val/NotCancerous'

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
        test_dir = '/voronoi_data/test'
    else:
        # Set the testing directory for local development
        test_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_data/test'

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
        cancer_test_dir = '/voronoi_data/test/Cancerous'
    else:
        # Set the cancerous testing directory for local development
        cancer_test_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_data/test/Cancerous'

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
        notCancer_test_dir = '/voronoi_data/test/NotCancerous'
    else:
        # Set the 'NotCancerous' testing directory for local development
        notCancer_test_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_data/test/NotCancerous'

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

def copy_directories_to_directories(dst_dir, dirs):
    """
    Copies a list of source directories into specified target directories based on their category.

    This function:
        - Iterates through the given list of directories.
        - Identifies whether each directory belongs to the "Cancerous" or "NotCancerous" category.
        - Copies each directory and its contents to the corresponding target directory under `dst_dir`.

    Args:
        dst_dir (str): The destination root directory where categorized directories will be copied.
        dirs (list of str): A list of source directory paths to be copied.

    Behavior:
        - Directories containing "NotCancerous" in their name are copied to `<dst_dir>/NotCancerous/`.
        - Directories containing "Cancerous" in their name are copied to `<dst_dir>/Cancerous/`.
        - If a directory does not match either category, it is ignored.

    Returns:
        None

    Side Effects:
        - Creates the necessary destination directories if they do not exist.
        - Copies directories and their contents using `shutil.copytree`.
        - Prints a confirmation message for each copied directory.
    """

    # Define the target directory paths for 'Cancerous' and 'NotCancerous' directories
    cancer_dir_path = dst_dir + '/Cancerous'
    not_cancer_dir_path = dst_dir + '/NotCancerous'

    # Iterate through each source directory in the provided list of directories
    for dir in dirs:
        # Convert the current directory to a Path object for easier handling
        source_dir = Path(dir)

        # Determine the appropriate target directory based on the name of the source directory
        if "NotCancerous" in str(dir):
            target_dir = not_cancer_dir_path  # Set the target for 'NotCancerous' directories
        elif "Cancerous" in str(dir):
            target_dir = cancer_dir_path  # Set the target for 'Cancerous' directories

        # Construct the full target path by joining the target directory with the source directory's name
        target_path = os.path.join(target_dir, os.path.basename(source_dir))

        # Copy the entire source directory, including its contents, to the target directory
        shutil.copytree(source_dir, target_path)

        # Print a confirmation message indicating successful directory copy
        print(f"Directory '{source_dir}' has been copied into '{target_dir}' as '{target_path}'.")

def define_train_transformer():
    """
    Defines and returns a transformation pipeline for preprocessing training images.

    The transformation includes:
        - Converting images to grayscale with a single output channel.
        - Random horizontal and vertical flipping to augment data and improve model generalization.
        - Applying random brightness adjustments while keeping contrast and saturation unchanged.
        - Converting images to a PyTorch tensor for deep learning model compatibility.
        - An optional reshaping step (commented out) for potential future adjustments.

    Returns:
        torchvision.transforms.Compose: A composed transformation to be applied to the training dataset.
    """

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
    """
    Defines and returns a transformation pipeline for preprocessing validation and testing images.

    The transformation includes:
        - Converting images to grayscale with a single output channel.
        - Converting images to a PyTorch tensor for compatibility with deep learning models.
        - An optional reshaping step (commented out) for potential future adjustments.

    Returns:
        torchvision.transforms.Compose: A composed transformation to be applied to validation and testing datasets.
    """

    # Create a composed transform for validation and testing dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert the image to grayscale with 1 output channel (removing color information)
        transforms.ToTensor(),                          # Convert the processed image to a PyTorch tensor for model input
        # transforms.Lambda(lambda x: x.reshape(1, 4608, 4608))  # (Optional) Reshape if needed; left commented for potential future use
    ])

    # Return the defined transformation for testing
    return transform

def sampler(dataset, count, size=1000):
    """
    Creates a weighted sampler for a given dataset to ensure balanced class representation
    during training by sampling underrepresented classes more frequently.

    This function:
    1. Counts the occurrences of each class in the dataset.
    2. Calculates class weights as the inverse of the class frequencies.
    3. Constructs a `WeightedRandomSampler` that samples from the dataset using the
       calculated class weights, allowing more frequent sampling of underrepresented classes.
    4. Returns the sampler for use with a DataLoader.

    Args:
        dataset (torch.utils.data.Dataset): The dataset from which to sample.
                                            It must have a `targets` attribute with class labels.
        count (int): The number of samples to consider from the dataset.
        size (int, optional): The number of samples to draw from the dataset (default is 1000).

    Returns:
        torch.utils.data.sampler.WeightedRandomSampler: A sampler that uses the calculated class weights
                                                        to sample data points with replacement.
    """

    # Count the occurrences of each class in the dataset using Counter
    class_counter = dict(Counter(dataset.targets))

    # Calculate class weights as the inverse of the class frequencies.
    # This gives higher weight to underrepresented classes,
    # ensuring they are sampled more frequently.
    class_weights = 1 / torch.Tensor([ct for ct in class_counter.values()])

    # # Create a list of indices for the training dataset.
    # # Use the count parameter to generate indices from 0 to count-1.
    dataset_indices = list(range(len(dataset)))

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

def splitting_data(image_patches, total_image_count, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, batch_size=32):
    """
    Splits the provided image patches into training, validation, and testing datasets,
    and returns DataLoader objects for each dataset.

    This function:
    1. Creates directories for storing training, validation, and testing data.
    2. Shuffles and sorts the image patches.
    3. Distributes image patches into respective datasets based on the given ratios.
    4. Copies image files into the appropriate directories for training, validation, and testing.
    5. Defines and applies transformations to the datasets.
    6. Creates DataLoader objects for training, validation, and testing datasets,
       with a weighted sampler for the training set to handle class imbalance.

    Args:
        image_patches (dict): A dictionary where the keys are image directories,
                              and the values are lists of image files in each directory.
        total_image_count (int): The total number of images to be split into datasets.
        train_ratio (float, optional): The proportion of images to be used for training (default is 0.68).
        val_ratio (float, optional): The proportion of images to be used for validation (default is 0.16).
        test_ratio (float, optional): The proportion of images to be used for testing (default is 0.16).
        batch_size (int, optional): The batch size to be used in the DataLoader (default is 32).

    Returns:
        tuple: A tuple containing three DataLoader objects:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - val_loader (DataLoader): DataLoader for the validation dataset.
            - test_loader (DataLoader): DataLoader for the testing dataset.
    """

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

    # Creates necessary directories for organizing training, validation, and testing data
    create_data_directories()

    # Copy the image files to their respective locations
    copy_directories_to_directories(train_dir, train_dirs)
    copy_directories_to_directories(val_dir, val_dirs)
    copy_directories_to_directories(test_dir, test_dirs)

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

class ModelEmbedding(nn.Module):
    '''
    A custom model wrapper that modifies a given model architecture by removing
    its final output layer and adding a linear layer for extracting embeddings.

    This class is designed to facilitate feature extraction by:
    1. Removing the model's original final layer.
    2. Adding a new linear layer to extract embeddings from the features.
    3. Including an additional final output layer for classification,
       allowing the model to be used for both embedding extraction and classification tasks.

    Attributes:
        features (nn.Sequential): A sequential container of layers excluding the final output layer.
        linear (nn.Linear): A linear layer for embedding extraction.
        relu (nn.ReLU): A ReLU activation function applied after the linear transformation.
        finlinear (nn.Linear): The final fully connected layer used for classification.
    '''

    def __init__(self, original_model):
        '''
        Initializes the ModelEmbedding by modifying the original model.

        Args:
            original_model (nn.Module): The pre-trained model whose final output layer
                                        will be removed for embedding extraction.
        '''

        # Initialize the parent class (nn.Module) to set up the module properly
        super(ModelEmbedding, self).__init__()

        # Remove the original model's final output layer
        self.features = nn.Sequential(*list(original_model.children()))[:-1]

        # Add a linear layer for embedding extraction
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.relu = nn.ReLU(inplace=True)

        # Add a new output layer with the appropriate number of output classes
        self.finlinear = nn.Linear(in_features=feature_dim, out_features=num_classes, bias=True)

    def forward(self, x):
        '''
        Defines the forward pass of the ModelEmbedding.

        This method processes the input through the feature extraction layers,
        applies embedding extraction, and then produces the final classification output.

        Args:
            x (torch.Tensor): The input tensor, typically a batch of images.

        Returns:
            tuple: A tuple containing:
                - embedding_out (torch.Tensor): The extracted embeddings after the linear transformation and ReLU.
                - out (torch.Tensor): The output from the final classification layer.
        '''

        # Pass input through the feature extraction layers of the original model
        embedding = self.features(x)

        # Reshape the embedding tensor to (batch_size, num_features) for compatibility with the linear layer
        embedding = embedding.view(embedding.size(0), -1)

        # Apply the linear layer and ReLU activation to extract the embedding
        embedding_out = self.relu(self.linear(embedding))

        # Pass the original embedding through the final output layer for classification
        out = self.finlinear(embedding)

        # Return both the extracted embedding and the final output
        return embedding_out, out

def load_model():
    """
    Loads a pre-trained ResNet18 model, modifies it for grayscale image input,
    and prepares it for training or inference with a custom final fully connected layer.

    The function performs the following operations:
    1. Loads the ResNet34 model pre-trained on ImageNet.
    2. Modifies the first convolutional layer (`conv1`) to accept grayscale images
       (1 input channel) by changing the input channel size and adjusting
       kernel, stride, and padding.
    3. Replaces the average pooling layer (`avgpool`) with an adaptive average pooling
       layer that outputs a size of (1, 1).
    4. Modifies the fully connected layer (`fc`) to match the number of classes (`num_classes`).
    5. Wraps the model in a custom `ModelEmbedding` class to extract embeddings
       by removing the final classification layer.
    6. Moves the model to a GPU if available, otherwise to a CPU for training or inference.

    Returns:
        tuple:
            - device (torch.device): The device to which the model is transferred (GPU/CPU).
            - model (torch.nn.Module): The modified ResNet34 model ready for use.

    """

    # Load the ResNet34 architecture pre-trained model
    # pretrained=True loads the model with pre-trained weights on ImageNet
    model = models.resnet18(pretrained=True)

    # Replace the first convolutional layer (conv1) of the model
    # in_channels=1: The number of input channels is set to 1 (for grayscale images)
    # out_channels=64: The number of output feature maps produced by this convolution layer
    # kernel_size=7: The size of the kernel (filter) used in this convolution operation is 7x7
    # stride=2: The stride (step size) the filter takes when moving across the input
    # padding=3: Zero-padding added around the input to preserve spatial dimensions
    # bias=False: No bias term is added in the convolution operation to reduce computational cost
    model.conv1 = nn.Conv2d(in_channels=1,
                            out_channels=64,
                            kernel_size=7,  # Kernel size is 7x7 for initial convolution
                            stride=2,
                            padding=3,  # Padding ensures the spatial size is preserved after convolution
                            bias=False)

    # Change the average pooling layer to adaptive pooling
    # nn.AdaptiveAvgPool2d(kernel_size=1) ensures the output is a fixed size (1x1)
    # regardless of the input size, useful for varying input sizes
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Output size is fixed to 1x1

    # Modify the final fully connected layer to match the number of classes
    # model.fc.in_features gives the input size to the fully connected layer
    # num_classes is the number of output classes (should be defined earlier in the code)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Wrap the modified model using the custom ModelEmbedding class
    # ModelEmbedding removes the final classification layer (model.fc)
    # and adds layers for embedding extraction (useful for transfer learning)
    model = ModelEmbedding(model)

    # Set the device to GPU if available, otherwise use CPU
    # This ensures the model runs on the available hardware (GPU if possible, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device (GPU or CPU) for training/inference
    model.to(device)  # Transfer the model to the specified device (GPU or CPU)

    return device, model  # Return the modified model and device for further use

def get_activation(name, activation):
    """
    Returns a hook function to capture the output (activation) of a specified layer in a neural network.

    This function is typically used to register forward hooks on layers of a PyTorch model, allowing
    access to intermediate activations during inference or training.

    Parameters:
    name (str): The name of the layer whose activation is being captured.
    activation (dict): A dictionary where the captured activations will be stored.
                       The key is `name`, and the value is the detached output tensor of the layer.

    Returns:
    function: A hook function that captures and stores the activation of the specified layer.

    Hook Function Behavior:
    - Takes three arguments: `model` (the layer/module), `input` (the layer input, not used),
      and `output` (the layer output).
    - Stores the detached output tensor in the `activation` dictionary under the given `name`.
    - Uses `.detach()` to prevent computation graph tracking and gradient updates.

    Example Usage:
    >>> activation_dict = {}
    >>> model.layer_name.register_forward_hook(get_activation("layer_name", activation_dict))

    After a forward pass, `activation_dict["layer_name"]` will contain the activations of `layer_name`.
    """

    def hook(model, input, output):
        # We store the output (activations) of the layer in the `activation` dictionary
        # The key is the `name` of the layer and the value is the detached output tensor.
        # `.detach()` is used to make sure that the tensor is not connected to the computation graph,
        # so it doesn't affect backpropagation and gradients.
        activation[name] = output.detach()

    # The `hook` function itself is returned by `get_activation`, so it can be registered.
    return hook

# originally 1e-5
def get_params(model, learningRate=0.001, weight_decay=0.01, factor=0.1, patience=10):
    """
    Initializes and returns the loss function, optimizer, and learning rate scheduler for training a model.

    Parameters:
        model (torch.nn.Module): The neural network model whose parameters will be optimized.
        learningRate (float, optional): The initial learning rate for the optimizer. Default is 1e-4.
        weight_decay (float, optional): The weight decay (L2 regularization) for the optimizer. Default is 1e-6.
        momentum (float, optional): Momentum factor (not applicable for AdamW but included for compatibility). Default is 0.7.
        factor (float, optional): The factor by which the learning rate is reduced when a plateau is detected. Default is 0.5.
        patience (int, optional): The number of epochs to wait before reducing the learning rate if no improvement is observed. Default is 3.

    Returns:
        tuple: A tuple containing:
            - criterion (torch.nn.CrossEntropyLoss): The loss function for multi-class classification.
            - optimizer (torch.optim.AdamW): The AdamW optimizer for training the model.
            - scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): The learning rate scheduler that adjusts the learning rate based on validation loss.

    """

    # Define the loss function as CrossEntropyLoss, commonly used for multi-class classification tasks.
    criterion = nn.CrossEntropyLoss()

    # Initialize the AdamW optimizer for the model's parameters with the specified learning rate.
    # The 'momentum' argument is not applicable for Adam but is included for consistency with other optimizers.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate, weight_decay=weight_decay)

    # Set up a learning rate scheduler that reduces the learning rate when a plateau in validation loss is detected.
    # The 'factor' specifies how much to reduce the learning rate.
    # The 'patience' defines how many epochs to wait before reducing the learning rate if no improvement is seen.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience)

    # Return the defined loss function, optimizer, and learning rate scheduler.
    return criterion, optimizer, scheduler

def train(model, device, train_loader, val_loader, criterion, optimizer, scheduler,
          num_epochs=5, start_epoch=0, all_train_embeddings=[], all_val_embeddings=[],
          all_train_loss=[], all_val_loss=[], all_train_acc=[], all_val_acc=[]):
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
    train_embeddings = []  # To store the extracted embeddings
    train_loss = []  # To store the training losses

    # Assuming num_classes is defined globally or passed as a parameter
    train_confusion_matrix = torch.zeros(num_classes, num_classes)  # Ensure num_classes is defined

    # Start training over the specified number of epochs
    for epoch in range(start_epoch, num_epochs):

        avg_loss = 0.0  # Initialize average loss for the current epoch

        losses = []
        accuracies = []

        # Iterate through batches of the training dataset
        for batch_num, (feats, labels) in enumerate(train_loader):

            # Reshape the features to match the input dimensions of the model
            feats = feats.reshape(-1, 1, feature_dim, feature_dim)  # Ensure feature_dim is defined
            feats, labels = feats.to(device), labels.to(device)  # Move to the specified device

            # Clear the gradients for the optimizer
            optimizer.zero_grad()

            # Pass the features through the model
            _, outputs = model(feats)

            # Get predicted labels by applying softmax and taking the maximum
            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)

            # Update the confusion matrix
            for t, p in zip(labels.view(-1), pred_labels):
                train_confusion_matrix[t.long(), p.long()] += 1

            # Assuming 'activation' is a defined variable within the model
            if 'avg_pool' in activation:
                train_embeddings.append(activation['avg_pool'].detach().cpu().numpy())  # Extract embeddings

            # Calculate the loss
            loss = criterion(outputs, labels.long())  # Compute the loss
            loss.backward()  # Backpropagate the loss

            # Update the model parameters
            optimizer.step()

            # Update accuracy for the current batch
            curr_accuracy = torch.sum(torch.eq(pred_labels, labels)).item()

            accuracy += curr_accuracy  # Increment correct predictions
            total += len(labels)  # Increment total number of samples

            # Store loss for each sample in the batch
            train_loss.extend([loss.item()] * feats.size(0))

            # Calculate average loss for the last 8 batches
            avg_loss += loss.item()
            if (batch_num + 1) % 8 == 0:
                print('Training Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch + 1, batch_num + 1, avg_loss / 8))
                avg_loss = 0.0  # Reset average loss after printing

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
        all_train_embeddings.extend(train_embeddings)
        all_train_loss.append(avg_train_loss)
        all_train_acc.append(avg_train_acc)

        # Assuming `epoch` is the variable that holds the current epoch number
        print(f'Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}, Average Training Accuracy: {avg_train_acc:.4f}')

        # Validate the model on the validation set
        val_confusion_matrix, val_embeddings, val_loss, val_acc = testing(model, device, val_loader, criterion)

        # Store validation metrics
        all_val_embeddings.extend(val_embeddings)
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

    return all_train_embeddings, all_val_embeddings, all_train_loss, all_val_loss, all_train_acc, all_val_acc

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
    val_embeddings = []  # List to store embeddings from the validation dataset

    # Initialize an empty confusion matrix to track model predictions vs true labels
    val_confusion_matrix = torch.zeros(num_classes, num_classes)

    # Iterate over validation data in batches
    for batch_num, (feats, labels) in enumerate(val_loader):

        # Reshape features into the required input shape (batch_size, channels, height, width)
        feats = feats.reshape(-1, 1, feature_dim, feature_dim)  # Reshape the input features to match the model's expected input dimensions
        feats, labels = feats.to(device), labels.to(device)  # Move the features and labels to the specified device (CPU or GPU)

        # Forward pass: Get the model's predictions (outputs)
        _, outputs = model(feats)

        # Get the predicted labels by finding the index of the maximum output value
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)  # Flatten the predicted labels tensor

        # Update the confusion matrix by comparing true labels with predicted labels
        for t, p in zip(labels.view(-1), pred_labels):
            val_confusion_matrix[t.long(), p.long()] += 1

        # Extract embeddings if 'avg_pool' is part of the activation dictionary
        if 'avg_pool' in activation:
            val_embeddings.append(activation['avg_pool'].detach().cpu().numpy())  # Store embeddings

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
    return val_confusion_matrix, val_embeddings, val_loss, val_acc

def create_results_directories():
    """
    Creates and sets up the necessary directories for storing results, including directories
    for accuracies and losses, depending on the execution environment (Docker or local).

    This function:
    1. Checks the environment to determine paths for the results, accuracies, and losses directories.
    2. Deletes existing directories (if present) before creating new ones to ensure the directories are fresh.
    3. Prints the status of the directory creation or deletion process.

    It handles the following directories:
        - Base results directory (`voronoi_results`)
        - Accuracies directory (`voronoi_results/accuracies`)
        - Losses directory (`voronoi_results/losses`)

    If running in a Docker environment, it uses the `/voronoi_results` directory, otherwise, it uses the local directory
    under `/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_results`.

    Returns:
        None
    """

    # Determine the base results directory based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the base results directory for the Docker environment
        results_dir = '/voronoi_results'
    else:
        # Set the base results directory for local development
        results_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_results'

    # Check if the base results directory exists and remove it if it does
    if os.path.exists(results_dir):
        # Remove the directory and all its contents if it exists
        shutil.rmtree(results_dir)
        print(f"Directory '{results_dir}' has been deleted.")
    else:
        # Notify if the directory does not exist
        print(f"Directory '{results_dir}' does not exist.")

    # Create the base results directory
    os.makedirs(results_dir, exist_ok=True)

    # Verify the creation of the base results directory and print the status
    if os.path.exists(results_dir):
        print(f"Directory '{results_dir}' created successfully.")
    else:
        # Notify if the directory creation failed
        print(f"Failed to create the directory '{results_dir}'.")

    # Determine the accuracies directory based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the accuracies directory for the Docker environment
        accuracies_dir = '/voronoi_results/accuracies'
    else:
        # Set the accuracies directory for local development
        accuracies_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_results/accuracies'

    # Create the accuracies directory
    os.makedirs(accuracies_dir, exist_ok=True)

    # Verify the creation of the accuracies directory and print the status
    if os.path.exists(accuracies_dir):
        print(f"Directory '{accuracies_dir}' created successfully.")
    else:
        # Notify if the directory creation failed
        print(f"Failed to create the directory '{accuracies_dir}'.")

    # Determine the losses directory based on the execution environment
    if os.path.exists('/.dockerenv'):
        # Set the losses directory for the Docker environment
        losses_dir = '/voronoi_results/losses'
    else:
        # Set the losses directory for local development
        losses_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_results/losses'

    # Create the losses directory
    os.makedirs(losses_dir, exist_ok=True)

    # Verify the creation of the losses directory and print the status
    if os.path.exists(losses_dir):
        print(f"Directory '{losses_dir}' created successfully.")
    else:
        # Notify if the directory creation failed
        print(f"Failed to create the directory '{losses_dir}'.")

def plotAccuracy(mode, accuracies):
    """
    Plots the accuracy of a model over time.

    Parameters:
    mode (str): A string indicating the type of accuracy being plotted (e.g., "Training", "Validation").
    accuracies (list): A list of accuracy values recorded over time.

    This function generates a plot that visualizes the accuracy values recorded during training
    or validation and saves it as a PNG image file at a specified path.
    """

    # Create titles for the graph and plot based on the mode (Training, Validation, etc.)
    graph_title = mode + " Accuracies"
    plot_title = mode + " Accuracy Over Time"

    # Define the file path where the plot will be saved
    save_path = "/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_results/accuracies/" + mode + "_accuracy_plot.png"

    # Generate time intervals corresponding to the number of recorded accuracy values
    time_intervals = list(range(1, len(accuracies) + 1))

    # Set up the figure size for the plot
    plt.figure(figsize=(8, 5))

    # Plot the accuracy values over time with markers
    plt.plot(time_intervals, accuracies, marker='o', linestyle='-', label=graph_title)

    # Label the x-axis to represent time intervals (epochs or iterations)
    plt.xlabel("Time Intervals")

    # Label the y-axis to represent the accuracy values at each interval
    plt.ylabel("Accuracy")

    # Set the plot title to reflect the mode (Training, Validation, etc.)
    plt.title(plot_title)

    # Add a legend to the plot to differentiate from other potential plots
    plt.legend()

    # Enable grid lines for better visualization of the plot
    plt.grid(True)

    # Save the plot as a PNG file at the specified save path
    plt.savefig(save_path)

def plotLoss(mode, losses):
    """
    Plots the loss of a model over time.

    Parameters:
    mode (str): A string indicating the type of loss being plotted (e.g., "Training", "Validation").
    losses (list): A list of loss values recorded over time.

    This function generates a plot that visualizes the loss values recorded during training
    or validation, saving it as a PNG image file in the specified path.
    """

    # Create titles for the graph and plot based on the mode (Training or Validation)
    graph_title = mode + " Losses"
    plot_title = mode + " Loss Over Time"

    # Define the file path where the plot will be saved
    save_path = "/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_results/losses/" + mode + "_loss_plot.png"

    # Generate time intervals corresponding to the recorded loss values
    time_intervals = list(range(1, len(losses) + 1))

    # Set up the figure size for the plot
    plt.figure(figsize=(8, 5))

    # Plot the loss values over time with markers
    plt.plot(time_intervals, losses, marker='o', linestyle='-', label=graph_title)

    # Label the x-axis as "Time Intervals" to represent the progression of epochs/iterations
    plt.xlabel("Time Intervals")

    # Label the y-axis as "Loss" to represent the loss value at each interval
    plt.ylabel("Loss")

    # Set the plot title to reflect the type of loss being plotted (Training or Validation)
    plt.title(plot_title)

    # Add a legend to differentiate this plot from others if multiple plots are displayed
    plt.legend()

    # Enable grid for better visualization of the plot
    plt.grid(True)

    # Save the plot as a PNG file at the specified save path
    plt.savefig(save_path)

def main():

    # Determine the patches directory based on the execution environment
    # Check if the code is running inside a Docker environment
    if os.path.exists('/.dockerenv'):
        cancer_patches_dir = '/voronoi_seg/Cancerous'  # Set the patches directory for Docker
        no_cancer_patches_dir = '/voronoi_seg/NotCancerous'
    else:
        cancer_patches_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_seg/Cancerous'  # Set the patches directory for local development
        no_cancer_patches_dir = '/Users/arpitha/Documents/Lab_Schwartz/code/imgFISH-nick/stardist/voronoi_seg/NotCancerous'

    # Initialize a dictionary to store image patches, categorized by subdirectory
    image_patches = {}

    # Counter to track the total number of image files across all directories
    total_image_count = 0

    # Retrieve image patches and update the total image count for the specified directories
    image_patches, total_image_count = get_patch_files(cancer_patches_dir, image_patches, total_image_count)
    image_patches, total_image_count = get_patch_files(no_cancer_patches_dir, image_patches, total_image_count)

    # Split the image patches into training, validation, and testing datasets according to specified ratios
    train_loader, val_loader, test_loader = splitting_data(image_patches, total_image_count)

    # Load the model architecture or pre-trained model.
    device, model = load_model()

    # Retrieve the hook function for 'avg_pool' layer to capture activations
    hook_function = get_activation('avg_pool', activation)

    # Register the hook to store activations from the specified layer during the forward pass
    model.features[8].register_forward_hook(hook_function)

    # Call the get_params function to obtain the loss function (criterion), optimizer, and learning rate scheduler for the model.
    criterion, optimizer, scheduler = get_params(model)

    # Train the model and retrieve the embeddings, training losses, and training accuracies
    all_train_embeddings, all_val_embeddings, all_train_loss, all_val_loss, all_train_acc, all_val_acc = train(model, device, train_loader, val_loader, criterion, optimizer, scheduler)

    # Run the testing phase of the model on the test dataset
    test_confusion_matrix, all_test_embeddings, all_test_loss, all_test_acc = testing(model, device, test_loader, criterion)

    # Print the confusion matrix for the testing phase
    print("Testing Confusion Matrix:\n", test_confusion_matrix)

    # Print the normalized confusion matrix to evaluate per-class accuracy
    # Normalization is done by dividing the diagonal by the sum of each row
    print("Testing Normalized Confusion Matrix (per-class accuracy):\n",
          test_confusion_matrix.diag() / test_confusion_matrix.sum(1))

    # Print the average testing loss and accuracy metrics
    print(f'Average Testing Loss: {all_test_loss:.4f}, Average Testing Accuracy: {all_test_acc:.4f}')

    # Create the necessary directories for saving results, including accuracies and losses
    create_results_directories()

    # Plot and save the accuracy values for the training, validation, and testing datasets
    plotAccuracy("training", all_train_acc)
    plotAccuracy("validation", all_val_acc)

    # Plot and save the loss values for the training, validation, and testing datasets
    plotLoss("training", all_train_loss)
    plotLoss("validation", all_val_loss)

    return all_train_acc, all_val_acc, all_test_acc  # Return lists of training, validation, and test accuracies

# Entry point for the script
if __name__ == "__main__":
    main()

# Doubts
  # 1) What size to do images because the layer size of resnet34 is 512? Images are blurry with size 512.
  # 2) If I have to look for more data what is recommended: Kaggle, NBIA Data Retriever, Pittsburgh Supercomputer?
