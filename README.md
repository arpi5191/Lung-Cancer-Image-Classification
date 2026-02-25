# Lung-Cancer-Image-Classification

A major challenge in medical research is the scarcity of imaging data. This project evaluates whether synthetic image data can potentially replace real and spatially segmented data by comparing how well a convolutional neural network (CNN) can perform cancer classification on them.

An HPC machine learning pipeline has been developed that sets up the environment, generates datasets, preprocesses them, trains a CNN model using CUDA, and evaluates classification performance using statistical testing. There were five datasets: real, spatially segmented (Voronoi diagrams), and three synthetic (generative AI). A latent diffusion model was built and trained on the original dataset to generate the synthetic images. For the other two datasets, a Stable Diffusion pipeline was fine-tuned with prompt/context engineering and LoRA/DoRA (parameter-efficient fine-tuning) on the original dataset and entirely from scratch.

If the CNN cancer classification performance on the synthetic datasets were equal to or better than that on the real and spatially segmented datasets, then synthetic images could potentially serve as an effective substitute in the absence of sufficient medical imaging data. This is a work in progress, and more details, results, and updates will be added to the README in future revisions.
