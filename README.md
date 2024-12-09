# Lung-Cancer-Image-Classification

Perform segmentation on the lung images and feed the resulting patches into Convolutional Neural Networks (CNNs) to classify whether they are cancerous or non-cancerous.

# Instructions:
  1) Run the command 'python seg.py --nmin nmin --nmax nmax --didx dapi --d downsizefactor' in the terminal to perform segmentation on the images in the tif folder. This will create a patch around
     each nucleus in the image and save it in the patches directory under the image folder.
  2) Run the command 'python ResNet50.py' in the terminal. This will perform data splitting and save the images into the training, validation, and testing folders under the data directory. It will
     then train a ResNet50 model on the dataset to classify the patches as cancerous or non-cancerous and generate evaluation metrics for the training, validation, and testing phases of the model.
  3) Optional: Run the command python CNNClassifier.py in the terminal, which performs the same task as ResNet50.py but uses a CNN model built from scratch. This CNN model did not work effectively
     on a similar dataset and is still subject to modifications.

# Note:
  If 'FileNotFoundError: Found no valid file for the classes Cancerous, NotCancerous. Supported extensions are: .jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp' error occurs when running
  python ResNet50, the data has not split properly, so rerun 'python ResNet50' in the terminal.

# References:
  https://github.com/CMUSchwartzLab/imgFISH/tree/nick/stardist
  https://github.com/CMUSchwartzLab/ExPath-CNN/tree/main
