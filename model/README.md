# Evaluating AI's Capability to Detect Brain Vessel Abnormalities Compared to Radiologist Review

Investigated how well AI models can identify abnormal blood vessel patterns in brain MRA scans compared to traditional radiologist review, applying Python, deep learning methods, and biomedical imaging concepts within AI4ALL's AI4ALL Ignite accelerator.

## Problem Statement

To what extent can AI detect abnormal vessel patterns in brain MRA scans compared to manual radiologist review?

Stroke prevention and early detection of vascular abnormalities rely heavily on accurate interpretation of brain MRA scans. Radiologists review these images manually, a process that is time-consuming and vulnerable to fatigue or human error. With the increasing volume of patients and imaging data, there is a growing need for tools that can support radiologists and improve diagnostic consistency. This project explores whether AI, specifically deep learning models, can classify healthy and abnormal vessel segments with meaningful accuracy. Understanding AI's strengths and weaknesses in this task has real impact on clinical workflows, second-reader systems, and neurovascular diagnostics.

## Key Results

1. Successfully trained a MONAI 3D ResNet-18 model achieving 81.4% recall on aneurysm detection
2. Processed and explored the VesselMNIST3D dataset containing 3D brain vessel segments (1,335 training, 191 validation, 382 test samples)
3. Identified and mitigated major dataset imbalance (8:1 ratio of healthy to aneurysm samples) using WeightedRandomSampler
4. Analyzed potential biases in the dataset:
   - Low image resolution (64×64×64 voxels) limiting detection of subtle vessel abnormalities
   - Unknown demographics reducing generalizability across diverse populations
   - Class imbalance requiring specialized training techniques
5. Optimized model performance through threshold adjustment, improving recall from 72% to 81.4%
6. Achieved clinically relevant metrics: 81.4% recall (sensitivity), 87.3% specificity, catching 35 out of 43 test aneurysms
7. Demonstrated model's viability as a screening/triage tool to assist radiologists in detecting life-threatening aneurysms

## Methodologies

To accomplish this, we utilized Python to load, preprocess, and visualize 3D vessel segments from the VesselMNIST3D dataset. We implemented comprehensive data preprocessing including normalization, channel dimension adjustment, and data augmentation (random flips, rotations, Gaussian noise, brightness/contrast adjustments) to enhance model robustness. We built a 3D Convolutional Neural Network using MONAI's ResNet-18 architecture, specifically designed for medical imaging applications. To address the 8:1 class imbalance, we employed WeightedRandomSampler during training to oversample the minority aneurysm class. The model was trained for 20 epochs using weighted cross-entropy loss, Adam optimizer, and learning rate scheduling. We evaluated performance using medical imaging metrics including accuracy, precision, recall, specificity, and F1-score. Finally, we applied threshold optimization to maximize recall (sensitivity) for aneurysm detection, prioritizing patient safety by minimizing false negatives at the cost of increased false positives that radiologists can review.

## Data Sources

- VesselMNIST3D Dataset: [MedMNIST](https://medmnist.com/)
- Intra: 3D Intracranial Aneurysm Dataset (Xi Yang et al., CVPR 2020)
- Related scientific literature provided through MedMNIST documentation and biomedical imaging research

## Technologies Used

- Python
- PyTorch
- MONAI (Medical Open Network for AI)
- NumPy
- pandas
- scikit-learn
- matplotlib
- 3D ResNet-18 (MONAI)
- Google Colab (T4 GPU)
- Jupyter Notebook
- MedMNIST tools and utilities

## Authors

This project was completed in collaboration with:
- Folabomi Longe
- Oluwatodimu Adegoke
- Ousman Bah
- Karen Maza Delgado
- Maria Garcia
- Chimin Liu
