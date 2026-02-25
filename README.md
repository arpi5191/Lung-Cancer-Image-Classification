# Lung Cancer Image Classification

## Project Overview

A major challenge in medical research is the scarcity of imaging data. This project investigates whether synthetic images can serve as effective substitutes for real and spatially segmented medical images by comparing cancer classification performance across image types using a convolutional neural network (CNN).

An HPC machine learning pipeline has been developed that sets up the environment, generates and preprocesses images, trains a CNN using CUDA, and evaluates classification performance via statistical testing. Five image sets are used: real scans, spatially segmented images (Voronoi diagrams), and three synthetic variants produced using generative AI. A latent diffusion model was built and trained on the original data to generate the first synthetic variant. The remaining two were produced by fine-tuning a Stable Diffusion pipeline using LoRA/DoRA (parameter-efficient fine-tuning) — one with prompt engineering and the other with context engineering.

If CNN classification performance on the synthetic images proves equal to or better than on the real and spatially segmented ones, synthetic data could serve as a viable substitute where sufficient medical imaging is unavailable. This is a work in progress — results and further details will be added in future revisions.

## File Descriptions
1) run_pipeline.slurm: Executes the machine learning pipeline. It sets up the environment, generates datasets, and preprocesses them. It then performs CNN cancer classification comparisons using compare_models.sh.
2) compare_models.sh: Runs the CNN model a specified number of times on each dataset and collects the test accuracies. It then applies Wilcoxon rank-sum tests to pairs of datasets using wilcoxon_test.py.
3) requirements.txt: Contains the packages that must be installed to set up the environment.
4) tumor_seg.py: Applies preprocessing techniques to the original lung histopathology image dataset.
5) voronoi_seg.py: Generates a voronoi diagram dataset from the preprocessed lung histopathology images
7) diffusion_seg.py: Generates a synthetic dataset by training a latent diffusion model on the preprocessed lung histopathology images.
8) prompt_seg.py: Generates a synthetic dataset by fine-tuning a Stable Diffusion pipeline using prompt engineering and LoRA/DoRA.
9) context_seg.py: Generates a synthetic dataset by fine-tuning a Stable Diffusion pipeline using context engineering and LoRA/DoRA.
10) extract_patches.py: Segments images in a dataset for input into the CNN model.
