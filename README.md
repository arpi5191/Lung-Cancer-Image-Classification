# Lung-Cancer-Image-Classification

## Project-Overview

A major challenge in medical research is the scarcity of imaging data. This project investigates whether synthetic images can serve as effective substitutes for real and spatially segmented images by comparing their cancer classification performance using a convolutional neural network (CNN).

An HPC machine learning pipeline has been developed that sets up the environment, generates datasets, preprocesses them, trains a CNN model using CUDA, and evaluates classification performance using statistical testing. There are five datasets: real, spatially segmented (Voronoi diagrams), and three synthetic (generative AI). A latent diffusion model was built and trained on the original data to generate the first synthetic dataset. For the other two datasets, a Stable Diffusion pipeline was fine-tuned using prompt/context engineering and LoRA/DoRA (parameter-efficient fine-tuning) on the original data and entirely from scratch.

If the CNN cancer classification performance on the synthetic datasets were equal to or better than that on the real and spatially segmented datasets, then synthetic images could potentially serve as an effective substitute in the absence of sufficient medical imaging data. This is a work in progress, and more details, results, and updates will be added to the README in future revisions.
