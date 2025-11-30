This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

# NeuroScan-Frontend: AI Diagnostic Assistant for Brain Vessel Abnormalities

This repository contains the Next.js frontend application for the NeuroScan project, which integrates with a Gradio backend running a 3D Convolutional Neural Network (3D-CNN) to classify brain vessel segments from MRA scans.

## Project Goal

The primary goal of the NeuroScan project is to **evaluate and deploy an AI model capable of detecting abnormal vessel patterns (aneurysms) in brain MRA scans.** This tool is intended to serve as a reliable second reader to assist radiologists in neurovascular diagnostics, specifically targeting critical, small vascular anomalies that are prone to human fatigue or oversight.

## System Architecture

The application uses a **three-part architecture** for robust deployment and scalability:

| Component | Technology | Responsibility | Status |
| :--- | :--- | :--- | :--- |
| **Frontend** | Next.js (TypeScript) | User interface for file upload, result display, and client-side logic (e.g., file renaming). | Deployed on Vercel |
| **Proxy Layer** | Next.js Rewrites (`next.config.js`) | Routes client-side API calls (`/api/gradio-proxy/`) securely to the external Gradio backend. | Configured & Active |
| **Backend/Model** | Gradio + Python (MONAI, PyTorch) | Hosts the 3D ResNet model, handles NIfTI file I/O, runs inference, and returns prediction results. | Hosted on Hugging Face Spaces |

---

## Frontend Setup (Next.js)

### Prerequisite

* Node.js (v18+)
* npm or yarn

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/KarenMazaDelgado/Neuroscan-Frontend.git](https://github.com/KarenMazaDelgado/Neuroscan-Frontend.git)
    cd Neuroscan-Frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run the development server:
    ```bash
    npm run dev
    ```
    The application will be accessible at `http://localhost:3000`.

### Key Frontend Files

* `next.config.js`: **CRITICAL!** Contains the proxy rewrite rule linking `/api/gradio-proxy/` to the Hugging Face Space.
* `app/page.tsx`: Contains the main component, the **`handleUpload`** function, which uses the `@gradio/client` library to connect to the backend. It also includes **client-side file-renaming logic** to prevent file-type errors.

---

## Machine Learning Model & Dataset

### Algorithm

* **Model:** **3D Convolutional Neural Network (3D-CNN)**, specifically a **3D ResNet model**.
* **Implementation:** The 3D ResNet is imported and used from the **MONAI** (Medical Open Network for AI) library, which is built on PyTorch and specialized for medical image analysis.
* **Technique:** Classification Modeling (Classifies vessel segments as **Normal** or **Aneurysm**).

### Dataset

* **Dataset:** **VesselMNIST3D** (a component of MedMNIST v2).
* **Input Data:** 3D volumetric images of size $28 \times 28 \times 28$ voxels.
* **Bias Note:** The data has a high proportion of healthy vessels to aneurysm segments, which may lead to the model favoring detection of healthy segments and being less effective on unhealthy ones (a form of data bias).
* **Preprocessing:** Input NIfTI files are resized to $28 \times 28 \times 28$ and normalized before inference.

---

