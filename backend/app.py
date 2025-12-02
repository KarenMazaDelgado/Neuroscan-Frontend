import gradio as gr
import torch
import numpy as np
import nibabel as nib
from skimage.transform import resize
from monai.networks.nets import resnet18
import os
import base64

# --- CONFIGURATION ---
TARGET_SIZE = (64, 64, 64)  # Match VesselMNIST3D training data
NUM_CLASSES = 2
CLASS_LABELS = ["Normal", "Aneurysm"]

# --- LOAD MODEL (Using MONAI) ---
model = resnet18(
    spatial_dims=3,
    n_input_channels=1,
    num_classes=NUM_CLASSES
)

# Load the weights
try:
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu'), weights_only=False))
    model.eval()
    print("Model loaded successfully from model.pth.")
except FileNotFoundError:
    print("ERROR: model.pth not found. Please ensure the file is present.")

# --- GRAD-CAM IMPLEMENTATION ---
class GradCAM3D:
    """
    Grad-CAM for 3D CNNs to generate attention heatmaps
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap

        Args:
            input_tensor: Input tensor [1, 1, D, H, W]
            target_class: Target class index (if None, uses predicted class)

        Returns:
            cam: 3D heatmap normalized to [0, 1]
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        output[0, target_class].backward()

        # Get gradients and activations
        gradients = self.gradients  # [1, C, D, H, W]
        activations = self.activations  # [1, C, D, H, W]

        # Global average pooling of gradients (importance weights)
        weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)  # [1, C, 1, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, D, H, W]

        # Apply ReLU (only positive influence)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

# Initialize Grad-CAM (target the last conv layer before pooling)
# For MONAI ResNet18, the last conv block is layer4
grad_cam = GradCAM3D(model, target_layer=model.layer4[-1].conv2)

# --- PREDICT FUNCTION ---
def predict(file_obj):
    """
    Processes a NIfTI file, runs inference, and returns:
    1. Classification results
    2. Original scan data (base64 encoded NIfTI)
    3. Heatmap data (base64 encoded NIfTI)
    """
    print(f"Received file: {file_obj}")
    print(f"File type: {type(file_obj)}")

    if hasattr(file_obj, '__dict__'):
        print(f"File attributes: {file_obj.__dict__}")

    if file_obj is None:
        print("ERROR: No file received")
        return {
            "predictions": [[label, 0.0] for label in CLASS_LABELS],
            "scan_data": None,
            "heatmap_data": None
        }

    # Handle different types of file_obj (could be string path or file object)
    if isinstance(file_obj, str):
        file_path = file_obj
    elif hasattr(file_obj, 'name'):
        file_path = file_obj.name
    else:
        print(f"ERROR: Unexpected file_obj type: {type(file_obj)}")
        return {
            "predictions": [[label, 0.0] for label in CLASS_LABELS],
            "scan_data": None,
            "heatmap_data": None
        }

    print(f"Extracted file path: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")

    # List directory contents to see what's actually there
    if not os.path.exists(file_path):
        dir_path = os.path.dirname(file_path)
        if os.path.exists(dir_path):
            print(f"Directory exists, contents: {os.listdir(dir_path)}")
        else:
            print(f"Directory doesn't exist: {dir_path}")

    # Validate file and add/fix extension
    try:
        # Check if file is actually gzipped by reading magic bytes
        with open(file_path, 'rb') as f:
            magic = f.read(2)
            is_gzipped = (magic == b'\x1f\x8b')
        print(f"File is gzipped: {is_gzipped}")

        lower_path = file_path.lower()
        has_nii_ext = lower_path.endswith('.nii')
        has_gz_ext = lower_path.endswith('.nii.gz')
        print(f"File extension check - ends with .nii: {has_nii_ext}, ends with .nii.gz: {has_gz_ext}")

        # Determine correct extension based on actual file content
        correct_ext = ".nii.gz" if is_gzipped else ".nii"

        # Case 1: File has .nii.gz but is not gzipped - fix it
        if has_gz_ext and not is_gzipped:
            new_path = file_path[:-3]  # Remove .gz
            os.rename(file_path, new_path)
            file_path = new_path
            print(f"Fixed incorrect .gz extension, renamed to: {file_path}")

        # Case 2: File has .nii but is gzipped - fix it
        elif has_nii_ext and is_gzipped and not has_gz_ext:
            new_path = file_path + ".gz"
            os.rename(file_path, new_path)
            file_path = new_path
            print(f"Added missing .gz extension, renamed to: {file_path}")

        # Case 3: File has no NIfTI extension - add correct one
        elif not has_nii_ext and not has_gz_ext:
            new_path = file_path + correct_ext
            os.rename(file_path, new_path)
            file_path = new_path
            print(f"Added extension {correct_ext}, renamed to: {file_path}")

        else:
            print(f"File extension is correct: {file_path}")

    except Exception as e:
        print(f"ERROR: File validation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "predictions": [[label, 0.0] for label in CLASS_LABELS],
            "scan_data": None,
            "heatmap_data": None
        }

    # Preprocessing and Inference
    try:
        # Check file header to see if it's actually a NIfTI file
        with open(file_path, 'rb') as f:
            header_bytes = f.read(4)
            print(f"File header (first 4 bytes): {header_bytes}")
            print(f"File header as hex: {header_bytes.hex()}")
            print(f"File size: {os.path.getsize(file_path)} bytes")

        # Load NIfTI data
        nifti_img = nib.load(file_path)
        img_data = nifti_img.get_fdata()
        original_shape = img_data.shape

        # Resize to target size (28x28x28) and Normalize
        img_data_resized = resize(img_data, TARGET_SIZE, mode='constant', anti_aliasing=True).astype(np.float32)

        if img_data_resized.max() > img_data_resized.min():
            img_data_normalized = (img_data_resized - img_data_resized.min()) / (img_data_resized.max() - img_data_resized.min())
        else:
            img_data_normalized = np.zeros(TARGET_SIZE, dtype=np.float32)

        # Convert to Tensor [Batch, Channel, Depth, Height, Width]
        inp = torch.from_numpy(img_data_normalized).float().unsqueeze(0).unsqueeze(0)

        print(f"Input tensor shape: {inp.shape}, min: {inp.min()}, max: {inp.max()}")

        # Run Inference
        with torch.no_grad():
            outputs = model(inp)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Format predictions
        results = []
        predicted_class = probs.argmax().item()
        for i, label in enumerate(CLASS_LABELS):
            results.append([label, float(probs[i])])

        print(f"Prediction result: {results}")
        print(f"Predicted class: {predicted_class} ({CLASS_LABELS[predicted_class]})")

        # Generate Grad-CAM heatmap for the predicted class
        # Need to create a new tensor that requires gradients
        inp_grad = torch.from_numpy(img_data_normalized).float().unsqueeze(0).unsqueeze(0)
        inp_grad.requires_grad = True
        heatmap = grad_cam.generate_cam(inp_grad, target_class=predicted_class)

        # Resize heatmap back to original scan size
        heatmap_resized = resize(heatmap, original_shape, mode='constant', anti_aliasing=True).astype(np.float32)

        # Create NIfTI images for scan and heatmap
        scan_nifti = nib.Nifti1Image(img_data.astype(np.float32), affine=nifti_img.affine)
        heatmap_nifti = nib.Nifti1Image(heatmap_resized, affine=nifti_img.affine)

        # Encode to base64 for transmission
        import tempfile
        def nifti_to_base64(nifti_image):
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                nib.save(nifti_image, tmp.name)
                tmp.flush()
                with open(tmp.name, 'rb') as f:
                    data = f.read()
                os.unlink(tmp.name)
                return base64.b64encode(data).decode('utf-8')

        scan_b64 = nifti_to_base64(scan_nifti)
        heatmap_b64 = nifti_to_base64(heatmap_nifti)

        return {
            "predictions": results,
            "scan_data": scan_b64,
            "heatmap_data": heatmap_b64,
            "predicted_class": CLASS_LABELS[predicted_class],
            "confidence": float(probs[predicted_class])
        }

    except Exception as e:
        print(f"MODEL/NIBABEL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "predictions": [[label, 0.0] for label in CLASS_LABELS],
            "scan_data": None,
            "heatmap_data": None
        }

# --- LAUNCH INTERFACE ---
iface = gr.Interface(
    fn=predict,
    inputs=gr.File(
        label="Upload .nii or .nii.gz 3D Volume",
        type="filepath"
    ),
    outputs=gr.JSON(label="Analysis Results"),
    title="NeuroScan 3D Aneurysm Classifier with Visualization",
    description="Upload a 3D NIfTI (.nii or .nii.gz) volume for classification (Normal vs. Aneurysm) with attention visualization.",
    api_name="predict"
)

iface.launch()
