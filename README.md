# NeuroScan: AI-Powered Brain Aneurysm Detection System

[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://neuroscan-frontend.vercel.app/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![License](https://img.shields.io/badge/license-Research-orange)](LICENSE)

**NeuroScan** is a full-stack medical AI application for detecting brain vessel abnormalities (aneurysms) from MRA scans. This system provides an AI-powered triage tool to assist radiologists in detecting potentially life-threatening brain aneurysms.

🌐 **[Live Demo](https://neuroscan-frontend.vercel.app/)**

*Investigated how well AI models can identify abnormal blood vessel patterns in brain MRA scans compared to traditional radiologist review, applying Python, deep learning methods, and biomedical imaging concepts within AI4ALL's AI4ALL Ignite accelerator.*

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Frontend](#-frontend)
- [Backend](#-backend)
- [Model Training](#-model-training--research)
- [Performance](#-current-performance)
- [Limitations](#-limitations--bias-analysis)
- [Future Work](#-future-improvements)
- [Authors](#-authors)
- [References](#-references)

---

## 🎯 Project Overview

### Problem Statement

**To what extent can AI detect abnormal vessel patterns in brain MRA scans compared to manual radiologist review?**

Stroke prevention and early detection of vascular abnormalities rely heavily on accurate interpretation of brain MRA scans. Radiologists review these images manually, a process that is:

- Time-consuming and vulnerable to fatigue or human error
- Affected by high patient volumes and imaging data overload
- Subject to inconsistency, especially for subtle findings

With the increasing volume of patients and imaging data, there is a growing need for tools that can support radiologists and improve diagnostic consistency. This project explores whether AI, specifically deep learning models, can classify healthy and abnormal vessel segments with meaningful accuracy. Understanding AI's strengths and weaknesses in this task has real impact on clinical workflows, second-reader systems, and neurovascular diagnostics.

### Clinical Context

- **6.5 million** people in the US have unruptured brain aneurysms
- **30,000** ruptures occur annually with **50% mortality rate**
- **226% increase** in radiologist errors during high-volume shifts (67-90 vs ≤19 studies)
- Sub-Saharan Africa has **<1 radiologist per 500,000 people**

### Our Solution

**NeuroScan addresses these challenges by:**

- ✅ Flagging potential aneurysms for clinical review (does not replace radiologist judgment)
- ✅ Reducing diagnostic burden by pre-screening scans during high-volume shifts
- ✅ Working with lower-resolution data (64×64×64 voxels) to support under-resourced healthcare settings
- ✅ Achieving **83.72% sensitivity** on validation set, approaching clinical MRA standards (~95%)
- ✅ Serving as a screening/triage tool to assist radiologists in detecting life-threatening aneurysms

---

## 📁 Repository Structure
```
Neuroscan-Frontend/
├── frontend/              # Next.js web application
│   ├── app/              # React components and pages
│   │   ├── page.tsx      # Main application logic
│   │   ├── components/
│   │   │   └── NiftiViewer.tsx  # 3D visualization
│   │   └── globals.css   # Global styles
│   ├── public/           # Static assets and test samples
│   │   └── test_samples_20.zip
│   ├── next.config.ts    # Backend proxy config
│   └── package.json      # Frontend dependencies
│
├── backend/              # Gradio inference server
│   ├── app.py           # Main inference script
│   ├── model.pth        # Trained model weights (133MB)
│   └── requirements.txt # Backend dependencies
│
├── model/               # Training code and experiments
│   ├── Project.ipynb   # Main training notebook
│   ├── DATA SET/       # VesselMNIST3D dataset
│   ├── RESULTS/        # Model checkpoints
│   └── VISUALS/        # Performance charts
│
└── README.md           # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/neuroscan-frontend.git
cd neuroscan-frontend
```

### 2. Start the Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Backend will run on `http://localhost:7860`

### 3. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend will run on `http://localhost:3000`

### 4. Test the System

1. Navigate to `http://localhost:3000`
2. Download test samples from the UI (20 jumbled scans included)
3. Upload a `.nii` or `.nii.gz` file
4. View predictions and 3D visualization with confidence heatmaps

---

## 🖥️ Frontend

### Features

- 🔍 **Single Scan Analysis** with interactive 3D visualization
- 📊 **Batch Processing** mode for multiple scans
- ⚖️ **Side-by-Side Comparison** mode
- 📥 **Download Test Samples** (20 jumbled aneurysm/normal scans)
- 🎨 **Real-Time Heatmap** visualization showing detection confidence
- 🏥 **Medical-Themed Interface** designed for clinical environments
- 📱 **Responsive Design** works on desktop and tablet devices

### Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Next.js** | 16 | React framework with server-side rendering |
| **React** | 19 | Component-based UI library |
| **TypeScript** | 5.x | Type-safe JavaScript |
| **Tailwind CSS** | 3.x | Utility-first CSS framework |
| **NIfTI.js** | Latest | Medical image file format parser |
| **WebGL** | 2.0 | Hardware-accelerated 3D graphics |
| **@gradio/client** | Latest | Backend API communication |
| **Vercel** | - | Deployment platform |

### Key Files
```
frontend/
├── app/
│   ├── page.tsx                    # Main application logic
│   │                                # - Upload handling
│   │                                # - Prediction display
│   │                                # - Batch processing
│   ├── components/
│   │   └── NiftiViewer.tsx        # 3D visualization component
│   │                                # - WebGL rendering
│   │                                # - Interactive controls
│   ├── layout.tsx                  # Root layout
│   └── globals.css                # Global styles & Tailwind
├── next.config.ts                 # Backend proxy config
├── tailwind.config.ts             # Tailwind customization
├── tsconfig.json                  # TypeScript config
└── public/
    └── test_samples_20.zip        # Test dataset
```

### Installation
```bash
cd frontend
npm install
npm run dev
```

### Environment Variables

Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:7860
NEXT_PUBLIC_APP_NAME=NeuroScan
```

### Build for Production
```bash
npm run build
npm start
```

### Deployment

The frontend is deployed on Vercel:
```bash
vercel deploy
```

---

## ⚙️ Backend

### Overview

Lightweight Python inference server that loads the trained 3D ResNet model and processes uploaded NIfTI files.

### Features

- 📁 Accepts `.nii` or `.nii.gz` files (medical imaging standard)
- 🔬 Returns binary classification (Aneurysm vs Normal) with confidence scores
- 🗺️ Generates attention heatmaps for visualization
- ☁️ Hosted on Hugging Face Spaces
- ⚡ Fast inference (~2-3 seconds per scan)

### Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Gradio** | 6.0.1 | Web interface framework |
| **PyTorch** | 2.0+ | Deep learning inference |
| **MONAI** | 1.3+ | Medical imaging toolkit |
| **nibabel** | Latest | NIfTI file processing |
| **NumPy** | 1.24+ | Numerical operations |
| **Pillow** | Latest | Image processing |

### Model Details

- **Architecture:** 3D ResNet-18 (MONAI)
- **Input:** 64×64×64 voxel 3D volumes (automatically resized)
- **Output:** Binary classification + confidence scores
- **Model Size:** 133MB
- **Parameters:** ~33 million
- **Inference Time:** ~2-3 seconds per scan (CPU)

### Installation
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### API Endpoints
```python
# Gradio Interface
predict(nifti_file) -> (prediction, confidence, heatmap)

# Returns:
# - prediction: "Aneurysm Detected" or "Normal (Healthy)"
# - confidence: float (0.0 to 1.0)
# - heatmap: PIL Image (attention visualization)
```

---

## 🧠 Model Training & Research

### Key Results

| Metric | Value | Clinical Significance |
|--------|-------|----------------------|
| **Recall (Sensitivity)** | 83.72% | Catches 36/43 aneurysms |
| **Specificity** | 87.3% | Correctly IDs healthy scans |
| **Precision** | 53.73% | 31 false positives |
| **Overall Accuracy** | 90.05% | Strong overall performance |
| **Aneurysms Detected** | 36/43 | 7 missed cases |
| **F1-Score** | 65.45% | Balanced metric |

### Methodology

**To accomplish this, we:**

1. **Data Preprocessing:**
   - Loaded and visualized 3D vessel segments from VesselMNIST3D dataset
   - Normalized pixel intensities from [0, 255] to [0, 1]
   - Adjusted channel dimensions for PyTorch/MONAI compatibility
   - Applied comprehensive data augmentation:
     - Random flips (horizontal, vertical, depth)
     - Random 90° rotations
     - Gaussian noise injection
     - Brightness/contrast adjustments

2. **Model Architecture:**
   - Built 3D Convolutional Neural Network using MONAI's ResNet-18
   - Specifically designed for medical imaging applications
   - Utilized residual connections to prevent vanishing gradients

3. **Training Strategy:**
   - Addressed 8:1 class imbalance using WeightedRandomSampler
   - Employed weighted cross-entropy loss (class weight: 7.5×)
   - Trained for 20 epochs with Adam optimizer
   - Applied learning rate scheduling
   - Used early stopping with patience of 5 epochs

4. **Evaluation:**
   - Evaluated using medical imaging metrics: accuracy, precision, recall, specificity, F1-score
   - Applied threshold optimization to maximize recall (sensitivity)
   - Prioritized patient safety by minimizing false negatives
   - Accepted increased false positives for radiologist review

5. **Model Comparison:**
   - Trained 5 different models (V1-V5) with varying class weights
   - Systematically compared performance metrics
   - Selected V5 as optimal balance of recall and precision

### Training Configuration
```python
# Model
architecture = "3D ResNet-18 (MONAI)"
input_shape = (64, 64, 64, 1)
output_classes = 2

# Training
loss_function = "Weighted Cross-Entropy"
class_weight = 7.5  # V5 configuration
optimizer = "Adam"
learning_rate = 1e-4
batch_size = 16
epochs = 20
early_stopping_patience = 5

# Hardware
device = "CUDA (T4 GPU)"
platform = "Google Colab"
```

### Model Versions Comparison

We trained 5 different models with varying class weights:

| Model | Class Weight | Accuracy | Recall | Precision | Missed | Notes |
|-------|-------------|----------|--------|-----------|--------|-------|
| V1 | 8.0× | 86.65% | 81.40% | 44.87% | 8 | Baseline |
| V2 | 12.0× | 89.53% | 83.72% | 52.17% | 7 | High recall |
| V3 | 9.5× | 90.58% | 81.40% | 55.56% | 8 | Best precision |
| V4 | 8.5× | 84.82% | 81.40% | 41.18% | 8 | Lower accuracy |
| **V5** | **7.5×** | **90.05%** | **83.72%** | **53.73%** | **7** | **✓ Best balance** |

**Why We Chose V5:**
- ✅ Tied for best recall (83.72%) - catches the most aneurysms
- ✅ Only 7 missed aneurysms (fewest false negatives)
- ✅ Strong precision (53.73%) - balanced false alarm rate
- ✅ High overall accuracy (90.05%)
- ✅ Optimal for clinical screening application

### Dataset

#### VesselMNIST3D

**Source:** 
- [MedMNIST v2](https://medmnist.com/)
- Original: IntrA: 3D Intracranial Aneurysm Dataset (Xi Yang et al., CVPR 2020)
- Related scientific literature provided through MedMNIST documentation

**Dataset Statistics:**
- **Total Samples:** 1,908
  - Training: 1,335 samples
  - Validation: 191 samples
  - Test: 382 samples
- **Resolution:** 64×64×64 voxels
- **Format:** Grayscale 3D volumes
- **Task:** Binary classification

**Class Distribution (Test Set):**
- Healthy: 339 samples (88.7%)
- Aneurysm: 43 samples (11.3%)
- **Imbalance Ratio:** 8:1

**Critical Challenge:**
Severe class imbalance caused initial models to predict "healthy" predominantly, achieving high accuracy but missing critical aneurysm cases.

### Challenges & Solutions

| Challenge | Impact | Solution Applied |
|-----------|--------|------------------|
| **Severe Class Imbalance (8:1)** | Model bias toward majority class | Weighted sampling + class weights (7.5×) + threshold optimization |
| **Small Image Resolution (64³)** | Limited detection of subtle features | Aggressive data augmentation + 3D CNN architecture |
| **Limited Training Data (43 aneurysms)** | Risk of overfitting | Early stopping + validation monitoring + regularization |
| **Inconsistent Results** | Performance varied between runs | Random seed setting + multiple training runs (V1-V5) |
| **Metric Selection** | Accuracy misleading with imbalance | Prioritized recall over accuracy (patient safety first) |

### Technologies Used

**Core Framework:**
- Python 3.8+
- PyTorch 2.0+
- MONAI 1.3+ (Medical Open Network for AI)

**Data Processing:**
- NumPy 1.24+
- pandas 2.0+
- nibabel (NIfTI file handling)
- scikit-learn 1.3+ (metrics, train/test split)

**Visualization:**
- matplotlib 3.7+
- seaborn 0.12+

**Development Environment:**
- Google Colab (T4 GPU)
- Jupyter Notebook
- MedMNIST tools and utilities

### Training Code
```bash
cd model
# Open Project.ipynb in Jupyter or Google Colab
# Ensure GPU runtime is enabled
# Runtime > Change runtime type > GPU (T4)
```

### Directory Structure
```
model/
├── Project.ipynb          # Main training notebook
│                          # - Data loading & EDA
│                          # - Model training (V1-V5)
│                          # - Evaluation & metrics
├── DATA SET/
│   └── vesselmnist3d.npz # Original dataset
├── RESULTS/
│   ├── model_v1.pth      # Model checkpoints
│   ├── model_v2.pth
│   ├── model_v3.pth
│   ├── model_v4.pth
│   └── model_v5.pth      # ✓ Best model (deployed)
├── VISUALS/
│   ├── training_curves.png    # Loss/accuracy plots
│   ├── confusion_matrix.png   # Classification matrix
│   ├── roc_curve.png          # ROC analysis
│   └── class_distribution.png # Dataset imbalance viz
└── test_samples/         # 20 anonymized test scans
```

---

## 📊 Current Performance

### Comparison with Clinical Standards

| Metric | NeuroScan (V5) | Clinical MRA Studies | Gap |
|--------|----------------|---------------------|-----|
| **Sensitivity (Recall)** | 83.72% | ~95% | -11.28% |
| **Specificity** | 87.3% | N/A | - |
| **Precision** | 53.73% | N/A | - |
| **Overall Accuracy** | 90.05% | N/A | - |
| **Resolution** | 64×64×64 voxels | 512×512×200+ voxels | -98% voxels |
| **Missed Aneurysms** | 7/43 (16.3%) | ~5% | +11.3% |
| **False Positives** | 31/339 (9.1%) | N/A | - |

### Performance Analysis

✅ **Strengths:**
- **Strong performance given constraints:** 83.72% recall with 64×64×64 resolution
- **High specificity (87.3%):** Reduces false alarm burden on radiologists
- **Viable screening tool:** Catches 36/43 aneurysms in test set
- **Patient safety priority:** Optimized to minimize missed life-threatening cases
- **Clinically relevant:** Approaching real-world MRA sensitivity standards

⚠️ **Performance Gap:**
- **11+ percentage point gap** from clinical MRA standards (~95% sensitivity)
- **Resolution limitation:** Primary bottleneck (64³ vs 512×512×200+)
- **7 missed aneurysms:** Would benefit from higher-quality training data
- **Room for improvement:** Architecture enhancements, ensemble methods

### Clinical Use Case

**Intended Role:** First-pass screening/triage tool
```
Patient MRA Scan
      ↓
NeuroScan Analysis
      ↓
   ┌──────┴──────┐
   │             │
High Risk    Low Risk
(Aneurysm)   (Normal)
   │             │
Radiologist  Standard
Priority     Review
Review       Queue
```

**Workflow Benefits:**
- 🎯 **Reduces workload** during high-volume shifts
- ⚡ **Flags obvious cases** for immediate radiologist attention
- 🔍 **Provides second opinion** for borderline cases
- 🌍 **Supports under-resourced** healthcare settings
- 📊 **Maintains transparency** with confidence scores

**Real-World Impact:**
- **9.1% false positive rate:** ~31 healthy scans flagged per 339 (manageable review burden)
- **16.3% false negative rate:** 7 missed aneurysms (improvement needed)
- **2-3 second inference:** Fast enough for real-time clinical integration

---

## 🔬 Limitations & Bias Analysis

### Identified Limitations

#### 1. Low Resolution (64×64×64 voxels)

**Impact:** Limits detection of subtle aneurysms

**Comparison:**
- NeuroScan: 64×64×64 voxels = 262,144 voxels
- Clinical MRA: 512×512×200+ voxels = 52+ million voxels
- **Information loss:** ~98% reduction in spatial detail

**Consequence:** Small or irregular aneurysms may be missed

#### 2. Unknown Demographics

**Missing Information:**
- Patient age, sex, ethnicity
- Geographic origin
- Hospital/scanner type
- Comorbidities
- Aneurysm risk factors

**Impact:** 
- Reduces generalizability across diverse populations
- May perform differently on underrepresented groups
- Unknown bias in training data distribution

#### 3. Class Imbalance (8:1 ratio)

**Challenge:**
- Only 43 aneurysm samples in test set (11.3%)
- 339 healthy samples (88.7%)

**Solutions Applied:**
- Weighted random sampling (oversample minority class)
- Class weights (7.5× penalty for false negatives)
- Threshold optimization (lowered from 0.5 to custom)
- Aggressive data augmentation

**Remaining Risk:**
- May not capture rare or atypical aneurysm presentations
- Performance may degrade on populations with different prevalence

#### 4. Single Imaging Modality

**Limitation:** Trained only on MRA scans

**Does NOT generalize to:**
- CTA (Computed Tomography Angiography)
- DSA (Digital Subtraction Angiography)
- Different MRI field strengths (1.5T vs 3T)
- Different scanner manufacturers
- Varying imaging protocols

#### 5. Binary Classification Only

**Current Capability:** Aneurysm vs Normal

**Missing Features:**
- Aneurysm size estimation (small/medium/large)
- Risk stratification (low/high rupture risk)
- Precise anatomical localization
- Multiple aneurysm detection
- Aneurysm type classification (saccular, fusiform, etc.)

### Potential Sources of Bias

| Bias Type | Source | Potential Impact | Mitigation Strategy |
|-----------|--------|------------------|---------------------|
| **Selection Bias** | Unknown patient demographics | May not generalize to all populations | Document limitation, recommend diverse validation |
| **Measurement Bias** | Low resolution (64³ voxels) | Systematic underdetection of small aneurysms | Acknowledge constraint, train on higher-res data |
| **Algorithmic Bias** | Class imbalance (8:1 ratio) | Model bias toward majority class | Weighted sampling, class weights, threshold tuning |
| **Deployment Bias** | Single modality (MRA only) | Won't work on CTA/DSA scans | Clearly scope intended use case |
| **Reporting Bias** | Cherry-picking metrics | Misleading performance claims | Report all metrics transparently |

### Bias Mitigation Strategies

✅ **What We Did:**

1. **Transparent Evaluation**
   - Reported ALL metrics (not just accuracy)
   - Tested 5 model versions (V1-V5)
   - Documented every limitation explicitly
   - Shared confusion matrix and error analysis

2. **Patient Safety Priority**
   - Optimized for high recall (83.72%)
   - Minimized false negatives (only 7 missed)
   - Accepted higher false positive rate (31 cases)
   - Clear communication: tool assists, not replaces radiologists

3. **Documented Limitations**
   - Resolution constraints (64³ vs 512×512×200+)
   - Demographic bias acknowledgment
   - Honest performance gap analysis (11% below clinical standard)
   - Explicit scope: screening/triage tool only

4. **Assistive Tool Design**
   - Designed to support (not replace) radiologists
   - Provides confidence scores for transparency
   - Enables human override at every step
   - Integrates into existing clinical workflow

5. **Systematic Approach**
   - Used established architectures (ResNet-18)
   - Applied best practices for imbalanced data
   - Validated across multiple training runs
   - Followed medical AI development guidelines

### Generalization Concerns

**Model trained on VesselMNIST3D may NOT generalize to:**

❌ Different MRI machines or scanning protocols  
❌ Different hospitals or healthcare systems  
❌ Populations with different aneurysm prevalence rates  
❌ Pediatric patients (if training data was adult-only)  
❌ Rare aneurysm subtypes not well-represented  
❌ Different ethnic/geographic populations  

**Recommendation:** Extensive external validation required before clinical deployment

---

## 🚧 Future Improvements

### Short-Term (3-6 months)

- [ ] Train on higher-resolution scans (128×128×128 voxels)
- [ ] Expand test dataset with more diverse samples
- [ ] Add aneurysm localization heatmaps (Grad-CAM)
- [ ] Implement confidence calibration techniques
- [ ] Create detailed error analysis dashboard
- [ ] Optimize inference speed for real-time use

### Medium-Term (6-12 months)

- [ ] Collect dataset with diverse patient demographics
- [ ] Test on multiple imaging modalities (MRA, CTA, DSA)
- [ ] External validation across 3+ hospitals
- [ ] Add risk stratification (size/severity classification)
- [ ] Develop explainability features (attention maps)
- [ ] Multi-class classification (aneurysm types)

### Long-Term (12+ months)

- [ ] Train on full-resolution clinical scans (512×512×200+)
- [ ] Conduct prospective clinical trials
- [ ] Integration with hospital PACS systems
- [ ] DICOM support and HL7 FHIR compatibility
- [ ] Real-time batch processing pipeline
- [ ] FDA approval pathway exploration

### Research Directions

**1. Architecture Improvements:**
- Vision Transformers (ViT) for 3D medical imaging
- Attention mechanisms for region focus
- Multi-scale feature fusion
- Ensemble methods (ResNet + DenseNet + ViT)
- 3D U-Net for precise segmentation

**2. Data Enhancements:**
- Synthetic data generation (GANs/Diffusion models)
- Semi-supervised learning with unlabeled scans
- Transfer learning from related tasks (vessel segmentation)
- Active learning for hard negative mining
- Few-shot learning for rare aneurysm types

**3. Clinical Integration:**
- DICOM file format support
- HL7 FHIR compatibility for EHR integration
- Batch processing pipeline for high-volume workflows
- Quality control metrics and monitoring
- Continuous learning from radiologist feedback

---

## 👥 Authors

This project was developed by:

- **Folabomi Longe** - [GitHub](https://github.com/FolabomiLonge) | [LinkedIn](https://linkedin.com/in/folabomi)
- **Ousman Bah** - [GitHub](https://github.com/ousmanbah10) | [LinkedIn](https://linkedin.com/in/ousman-bah)
- **Karen Maza Delgado** - [GitHub] (https://github.com/KarenMazaDelgado/) | [LinkedIn](https://www.linkedin.com/in/karenmaza/)
- **Maria Garcia** - 
- **Chimin Liu** - [GitHub](https://github.com/cooleschimo) | [LinkedIn](https://linkedin.com/in/chimin-liu)

*This project was completed in collaboration with the team as part of the **[AI4ALL Ignite](https://ai-4-all.org/)** accelerator program, investigating AI's capability to detect brain vessel abnormalities compared to radiologist review.*

---

## 📚 References

### Dataset & Model

1. Yang, X., et al. (2020). "IntrA: 3D Intracranial Aneurysm Dataset for Deep Learning." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_IntrA_3D_Intracranial_Aneurysm_Dataset_for_Deep_Learning_CVPR_2020_paper.pdf)

2. Yang, J., et al. (2023). "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification." *Scientific Data*. [Link](https://medmnist.com/)

3. MONAI Consortium. (2020). "MONAI: Medical Open Network for AI." [Documentation](https://monai.io/)

### Clinical Context

4. Hanna, T. N., et al. (2018). "The Effects of Fatigue from Overnight Shifts on Radiology Search Patterns and Diagnostic Performance." *Radiology*, 287(1), 91-98. [Paper](https://pubs.rsna.org/doi/10.1148/radiol.2017170900)

5. Ivanovic, V., et al. (2024). "Increased Study Volumes Are Associated with Increased Error Rates in Neuroradiology." *American Journal of Neuroradiology*. [Paper](https://www.ajnr.org/)

6. Vlak, M. H., et al. (2011). "Prevalence of unruptured intracranial aneurysms, with emphasis on sex, age, comorbidity, country, and time period: a systematic review and meta-analysis." *The Lancet Neurology*, 10(7), 626-636.

### Related Work

7. Ueda, D., et al. (2019). "Deep Learning for MR Angiography: Automated Detection of Cerebral Aneurysms." *Radiology*, 290(1), 187-194.

8. Timmins, K. M., et al. (2021). "Comparing methods of detecting and segmenting unruptured intracranial aneurysms on TOF-MRAs: The ADAM challenge." *NeuroImage*, 238, 118216.

---

## ⚖️ License

This project is licensed under a research-only license. The VesselMNIST3D dataset is based on the IntrA dataset (Xi Yang et al., CVPR 2020) and is used under the terms specified by MedMNIST.

### Usage Restrictions

- ✅ Academic research
- ✅ Educational purposes
- ✅ Non-commercial experimentation
- ❌ Clinical deployment without validation
- ❌ Commercial use without permission

---

## ⚠️ Disclaimer

**IMPORTANT: This tool is for research and experimental purposes only.**

- ⚠️ All predictions must be verified by qualified medical professionals
- ⚠️ NeuroScan is designed to complement, not replace, clinical judgment
- ⚠️ Not FDA approved or clinically validated
- ⚠️ Not intended for diagnostic use
- ⚠️ Performance may vary across different populations and imaging protocols

**Medical professionals should:**
- Use as a screening/triage tool only
- Independently review all flagged cases
- Not rely solely on AI predictions for patient care decisions
- Report any observed performance issues or biases

---


## 🙏 Acknowledgments

- **AI4ALL** for providing the Ignite accelerator program and guidance throughout the project

