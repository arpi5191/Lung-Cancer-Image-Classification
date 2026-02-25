# Lung Cancer Image Classification

## Project Overview

A major challenge in medical research is the scarcity of imaging data. This project investigates whether synthetic images can serve as effective substitutes for real and spatially segmented images by comparing their cancer classification performance using a convolutional neural network (CNN).

An HPC machine learning pipeline has been developed that sets up the environment, generates datasets, preprocesses them, trains a CNN model using CUDA, and evaluates classification performance using statistical testing. There are five datasets: real, spatially segmented (Voronoi diagrams), and three synthetic (generative AI). A latent diffusion model was built and trained on the original data to generate the first synthetic dataset. For the other two datasets, a Stable Diffusion pipeline was fine-tuned using LoRA/DoRA (parameter-efficient fine-tuning), with one dataset using prompt engineering and the other using context engineering.

If the CNN cancer classification performance on the synthetic datasets were equal to or better than that on the real and spatially segmented datasets, then synthetic images could potentially serve as an effective substitute in the absence of sufficient medical imaging data. This is a work in progress, and more details, results, and updates will be added to the README in future revisions.

## File Descriptions
1) run_pipeline.slurm: Executes the machine learning pipeline. It sets up the environment, generates datasets, and preprocesses them. It then performs CNN cancer classification comparisons using compare_models.sh.
2) compare_models.sh: Runs the CNN model a specified number of times on each dataset and collects the test accuracies. It then applies Wilcoxon rank-sum tests to pairs of datasets using wilcoxon_test.py.
3) requirements.txt: Contains the packages that must be installed to set up the environment.
4) tumor_seg.py: Applies preprocessing techniques to the original lung histopathology image dataset.
5) voronoi_seg.py: Generates a voronoi diagram dataset from the preprocessed lung histopathology images
7) diffusion_seg.py: Generates a synthetic dataset by training a latent diffusion model on the preprocessed lung histopathology images.
8) prompt_seg.py: Generates a synthetic dataset by fine-tuning a Stable Diffusion pipeline
9)
10) 
