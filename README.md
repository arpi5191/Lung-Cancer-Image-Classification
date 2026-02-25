# Lung Cancer Image Classification

## Project Overview

A major challenge in medical research is the scarcity of imaging data. This project investigates whether synthetic images can serve as effective substitutes for real and spatially segmented medical images by comparing cancer classification performance across image types using a convolutional neural network (CNN).

An HPC machine learning pipeline has been developed that sets up the environment, generates and preprocesses images, trains a CNN using CUDA, and evaluates classification performance via statistical testing. Five image sets are used: real scans, spatially segmented images (Voronoi diagrams), and three synthetic variants produced using generative AI. A latent diffusion model was built and trained on the original data to generate the first synthetic variant. The remaining two were produced by fine-tuning a Stable Diffusion pipeline using LoRA/DoRA (parameter-efficient fine-tuning) — one with prompt engineering and the other with context engineering.

If CNN classification performance on the synthetic images proves equal to or better than on the real and spatially segmented ones, synthetic data could serve as a viable substitute where sufficient medical imaging is unavailable. 

An additional aim was to explore whether nuclei features alone could predict malignancy. Voronoi regions were generated from the original dataset and handcrafted features were extracted and compiled into a dataframe. Traditional machine learning classifiers were then applied to predict whether the corresponding tissue sample was malignant or benign.

This is a work in progress — results and further details will be added in future revisions.

## File Descriptions

1) run_pipeline.slurm: Executes the machine learning pipeline. It sets up the environment, generates datasets, and preprocesses them. It then performs CNN cancer classification comparisons using compare_models.sh.
2) requirements.txt: Contains the packages that must be installed to set up the environment.
3) tumor_seg.py: Applies preprocessing techniques to the original lung histopathology image dataset.
4) voronoi_seg.py: Generates a voronoi diagram dataset from the preprocessed lung histopathology images
5) diffusion_seg.py: Generates a synthetic dataset by training a latent diffusion model on the preprocessed lung histopathology images.
6) prompt_seg.py: Generates a synthetic dataset by fine-tuning a Stable Diffusion pipeline using prompt engineering and LoRA/DoRA.
7) context_seg.py: Generates a synthetic dataset by fine-tuning a Stable Diffusion pipeline using context engineering and LoRA/DoRA.
8) extract_patches.py: Segments images in a dataset for input into the CNN model.
10) compare_models.sh: Runs the CNN model a specified number of times on each dataset and collects the test accuracies. It then applies Wilcoxon rank-sum tests to pairs of datasets using wilcoxon_test.py.
11) ResNetModel.py: Runs the CNN model on a specified dataset and returns the cancer classification test accuracy, executed on a high-performance computing environment.
12) ResNetModel_local.py: Runs the CNN model on a specified dataset and returns the cancer classification test accuracy, executed on a local environment.
13) wilcoxon_test.py: Runs the Wilcoxon rank-sum test on a pair of cancer classification accuracies, returning the p-value.
14) voronoi.py: Generates Voronoi regions, extracts features from them, and applies traditional machine learning classifiers to classify tissue samples as malignant.

## Usage

To run the machine learning pipeline, execute run_pipeline.slurm in a high-performance computing environment. The results can be viewed in resnet_out.txt.

## References

1) https://github.com/CMUSchwartzLab/imgFISH/tree/nick/stardist
2) https://github.com/CMUSchwartzLab/ExPath-CNN/tree/main
