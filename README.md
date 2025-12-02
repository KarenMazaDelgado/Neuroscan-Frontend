# NeuroScan: AI-Powered Brain Aneurysm Detection System

**NeuroScan** is a full-stack medical AI application for detecting brain vessel abnormalities (aneurysms) from MRA scans. This repo contains the complete system: frontend interface, inference backend, and model training code.

## Project Overview

**Problem Statement:** To what extent can AI detect abnormal vessel patterns in brain MRA scans compared to manual radiologist review?

NeuroScan serves as an AI-powered triage tool to assist radiologists in detecting potentially life-threatening brain aneurysms. The system:

- Flags potential aneurysms for clinical review (does not replace radiologist judgment)
- Reduces diagnostic burden by pre-screening scans during high-volume shifts
- Works with lower-resolution data (64×64×64 voxels) to support under-resourced healthcare settings
- Achieves 83.72% sensitivity on the validation set, approaching the ~95% sensitivity reported in clinical MRA studies

## Repository Structure

This repo contains three main components:

```
Neuroscan-Frontend/
├── frontend/          # Next.js web application (TypeScript/React)
├── backend/           # Gradio inference server (Python/PyTorch)
├── model/             # Training code and experiments (Jupyter/MONAI)
└── README.md          # This file
```

---

## Frontend (Next.js)

### Overview
Professional medical-themed interface for uploading brain MRA scans (NIfTI format), viewing predictions, and exploring model performance.

### Features
- Single scan analysis with 3D visualization
- Batch processing mode for multiple scans
- Side-by-side comparison mode
- Download test samples (20 jumbled scans included)
- Real-time heatmap visualization showing detection confidence

### Tech Stack
- **Framework:** Next.js 16 (React 19, TypeScript)
- **Styling:** Tailwind CSS
- **3D Visualization:** NIfTI.js, WebGL
- **API Client:** @gradio/client

### Setup

```bash
cd frontend
npm install
npm run dev  # Runs on http://localhost:3000
```

### Key Files
- `app/page.tsx` - Main application component with upload/prediction logic
- `app/components/NiftiViewer.tsx` - 3D brain scan visualization
- `next.config.ts` - Proxy configuration for backend API
- `public/test_samples_20.zip` - Test dataset with jumbled aneurysm/normal scans

---

## Backend (Gradio + PyTorch)

### Overview
Lightweight Python inference server that loads the trained 3D ResNet model and processes uploaded NIfTI files.

### Features
- Accepts `.nii` or `.nii.gz` files
- Returns classification predictions (Aneurysm vs Normal)
- Generates attention heatmaps for visualization
- Hosted on Hugging Face Spaces

### Tech Stack
- **Framework:** Gradio 6.0.1
- **Deep Learning:** PyTorch, MONAI
- **Image Processing:** nibabel, NumPy

### Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py  # Runs on http://localhost:7860
```

### Model
- **Architecture:** 3D ResNet-18 (MONAI)
- **Input:** 64×64×64 voxel 3D volumes (automatically resized)
- **Output:** Binary classification (Aneurysm/Normal) + confidence scores
- **Weights:** `model.pth` (133MB)

---

## Model Training & Research

### Overview
Research code for training and evaluating the 3D CNN on the VesselMNIST3D dataset. This work was completed as part of the **AI4ALL Ignite accelerator**.

### Key Results
- **81.4% recall** (sensitivity) on aneurysm detection
- **87.3% specificity**
- Successfully detected **35 out of 43** test aneurysms
- Addressed 8:1 class imbalance using WeightedRandomSampler
- Optimized threshold to minimize false negatives (patient safety priority)

### Limitations Identified
- **Low resolution** (64×64×64 voxels) limits detection of subtle aneurysms
- **Unknown demographics** in dataset reduces generalizability
- **Class imbalance** requires careful training strategies
- **11+ percentage point gap** compared to clinical MRA (~95% sensitivity)

### Dataset
- **Source:** VesselMNIST3D from MedMNIST v2
- **Original:** Intra: 3D Intracranial Aneurysm Dataset (Xi Yang et al., CVPR 2020)
- **Size:** 1,335 training / 191 validation / 382 test samples
- **Imbalance:** ~8:1 ratio (healthy:aneurysm)

### Tech Stack
- Python, PyTorch, MONAI
- NumPy, pandas, scikit-learn
- matplotlib for visualization
- Google Colab (T4 GPU)

### Training Code
```bash
cd model
# Open Project.ipynb in Jupyter or Google Colab
```

### Data
- `test_samples/` - 20 anonymized test scans for validation
- `DATA SET/` - Original VesselMNIST3D files
- `RESULTS/` - Model checkpoints and metrics
- `VISUALS/` - Training curves and performance charts

---

## Quick Start (Full Stack)

### 1. Start the Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py  # Runs on http://localhost:7860
```

### 2. Start the Frontend
```bash
cd frontend
npm install
npm run dev  # Runs on http://localhost:3000
```

### 3. Test the System
1. Download test samples from the frontend UI
2. Upload a `.nii` file
3. View predictions and 3D visualization

---

## Current Performance

| Metric | Value | Clinical MRA Studies |
|--------|-------|---------------------|
| **Sensitivity (Recall)** | 83.72% | ~95% |
| **Specificity** | 87.3% | N/A |
| **Resolution** | 64×64×64 voxels | 512×512×200+ voxels |

**Performance Gap:** While comparatively accurate given the resolution constraints, there remains an 11+ percentage point gap compared to clinical standards.

**Room for Improvement:**
- Higher-resolution training data
- Expanded dataset with diverse demographics
- Architectural refinements (attention mechanisms, multi-scale processing)
- Ensemble methods

---

## Research Background

### Problem Statement
To what extent can AI detect abnormal vessel patterns in brain MRA scans compared to manual radiologist review?

Stroke prevention and early detection of vascular abnormalities rely heavily on accurate interpretation of brain MRA scans. Radiologists review these images manually, a process that is:
- Time-consuming and vulnerable to fatigue
- Affected by high patient volumes and long shifts
- Subject to human error, especially for subtle findings

This project explores whether AI can provide meaningful assistance in this critical diagnostic task.

### Clinical Context
- 6.5 million people in the US have unruptured brain aneurysms
- 30,000 ruptures occur annually
- 50% mortality rate once ruptured
- Radiologist fatigue increases errors by 226% at high shift volumes (67-90 vs ≤19 studies)
- Sub-Saharan Africa has <1 radiologist per 500,000 people

NeuroScan addresses these challenges by providing automated pre-screening that flags potential aneurysms for clinical review, reducing the cognitive burden on radiologists during high-volume shifts. By working with lower-resolution data and supporting older 1.5T MRI systems, the tool extends advanced screening capabilities to under-resourced healthcare settings where specialist access is limited. While not replacing expert judgment, NeuroScan serves as a consistent second reader that can help maintain diagnostic quality regardless of workload or fatigue, potentially catching cases that might otherwise be missed during long shifts or in settings with limited radiologist availability.

---

## Authors

This project was developed by:
- Folabomi Longe
- Oluwatodimu Adegoke
- Ousman Bah
- Karen Maza Delgado
- Maria Garcia
- Chimin Liu

Completed as part of the AI4ALL Ignite accelerator program, investigating AI's capability to detect brain vessel abnormalities compared to radiologist review.

---

## License

This project uses the VesselMNIST3D dataset from MedMNIST, which is based on the IntrA dataset (Xi Yang et al., CVPR 2020).

## Disclaimer

This tool is for research and experimental purposes only. All predictions must be verified by qualified medical professionals. NeuroScan is designed to complement, not replace, clinical judgment.

---

## References

- MedMNIST v2: https://medmnist.com/
- IntrA Dataset: Xi Yang et al., "IntrA: 3D Intracranial Aneurysm Dataset for Deep Learning", CVPR 2020
- MONAI: Medical Open Network for AI - https://monai.io/
- Radiologist cognitive fatigue: Hanna et al., Radiology 2018
- Error rate volume correlation: Ivanovic et al., AJNR 2024
