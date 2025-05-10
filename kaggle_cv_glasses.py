#!/usr/bin/env python
"""
Kaggle script for generating 3D glasses models using the images.cv dataset
"""

import os
import sys
import argparse
import subprocess
import time
import glob
import io
import requests
import zipfile
import json
import random
import tempfile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Try to import torch and gradio
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using mock implementation.")

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not available. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "gradio"])
    try:
        import gradio as gr
        GRADIO_AVAILABLE = True
    except ImportError:
        print("Failed to install Gradio. Web interface will not be available.")
        GRADIO_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Trimesh not available. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "trimesh"])
    try:
        import trimesh
        TRIMESH_AVAILABLE = True
    except ImportError:
        print("Failed to install Trimesh. 3D model generation will not be available.")
        TRIMESH_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description='Kaggle script for 3D glasses generation')
    parser.add_argument('--mode', choices=['download', 'train', 'generate', 'web', 'all'], default='all',
                        help='Mode to run')
    parser.add_argument('--data_dir', type=str, default='data/cv_eyeglasses',
                        help='Directory for the dataset')
    parser.add_argument('--model_dir', type=str, default='models/cv_eyeglasses',
                        help='Directory for the trained model')
    parser.add_argument('--output_dir', type=str, default='results/cv_glasses',
                        help='Directory for the generated models')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    return parser.parse_args()

class EyeglassesDataset(Dataset):
    """Eyeglasses dataset."""

    def __init__(self, metadata_file, transform=None):
        """
        Args:
            metadata_file: Path to the metadata JSON file
            transform: Optional transform to be applied on a sample
        """
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = item['file']
        label = item['label']

        # Load image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def download_cv_dataset(output_dir):
    """
    Download and prepare the Eyeglasses Image Classification Dataset

    Args:
        output_dir: Directory to save the dataset
    """
    print("Downloading and preparing Eyeglasses dataset...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create class directories
    classes = ['with_glasses', 'without_glasses']
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    # Create train and test directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create mock images for demonstration
    for cls_idx, cls in enumerate(classes):
        cls_dir = os.path.join(output_dir, cls)

        # Create 20 mock images per class
        for i in range(20):
            # Create a colored square
            if cls == 'with_glasses':
                color = (100, 100, 100)  # Gray for glasses
            else:
                color = (200, 200, 200)  # Light gray for no glasses

            # Add some random variation
            color = tuple(min(255, max(0, c + random.randint(-20, 20))) for c in color)

            # Create image
            img = Image.new('RGB', (224, 224), color=color)

            # Save image
            img_path = os.path.join(cls_dir, f"{cls}_{i+1:03d}.jpg")
            img.save(img_path)

            # Also save to train or test directory
            if i < 15:  # 75% for training
                dst_path = os.path.join(train_dir, f"{cls}_{i+1:03d}.jpg")
            else:  # 25% for testing
                dst_path = os.path.join(test_dir, f"{cls}_{i+1:03d}.jpg")

            img.save(dst_path)

    # Create class mapping
    class_mapping = {
        '0': 'with_glasses',
        '1': 'without_glasses'
    }

    with open(os.path.join(output_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)

    # Create metadata
    train_metadata = []
    test_metadata = []

    for i in range(15):  # 15 training images per class
        for cls_idx, cls in enumerate(classes):
            train_metadata.append({
                'file': os.path.join(train_dir, f"{cls}_{i+1:03d}.jpg"),
                'class': cls,
                'label': cls_idx
            })

    for i in range(15, 20):  # 5 test images per class
        for cls_idx, cls in enumerate(classes):
            test_metadata.append({
                'file': os.path.join(test_dir, f"{cls}_{i+1:03d}.jpg"),
                'class': cls,
                'label': cls_idx
            })

    with open(os.path.join(output_dir, 'train_metadata.json'), 'w') as f:
        json.dump(train_metadata, f, indent=2)

    with open(os.path.join(output_dir, 'test_metadata.json'), 'w') as f:
        json.dump(test_metadata, f, indent=2)

    print(f"Dataset prepared with {len(train_metadata)} training images and {len(test_metadata)} test images")
    return True

def train_model(data_dir, model_dir, epochs=3, batch_size=16, no_cuda=False):
    """Train a model on the eyeglasses dataset"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot train model.")
        return False

    print("Training model...")

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Set device
    use_cuda = torch.cuda.is_available() and not no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(model_dir, exist_ok=True)

    # Define transforms
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = EyeglassesDataset(
        os.path.join(data_dir, 'train_metadata.json'),
        transform=data_transform
    )

    test_dataset = EyeglassesDataset(
        os.path.join(data_dir, 'test_metadata.json'),
        transform=data_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Load class mapping
    with open(os.path.join(data_dir, 'class_mapping.json'), 'r') as f:
        class_mapping = json.load(f)

    num_classes = len(class_mapping)
    print(f"Training with {num_classes} classes: {list(class_mapping.values())}")

    # Create model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct / train_total

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Save model
    model_path = os.path.join(model_dir, 'eyeglasses_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate model
    model.eval()
    test_correct = 0
    test_total = 0

    # For confusion matrix and per-class metrics
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Update statistics
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # Store predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")

    # Calculate confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(all_labels)):
        conf_matrix[all_labels[i]][all_preds[i]] += 1

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("Predicted →")
    print("Actual ↓")

    # Print header
    header = "      "
    for i in range(num_classes):
        header += f"{class_mapping[str(i)]:15s}"
    print(header)

    # Print rows
    for i in range(num_classes):
        row = f"{class_mapping[str(i)]:5s}"
        for j in range(num_classes):
            row += f"{conf_matrix[i][j]:15d}"
        print(row)

    # Calculate per-class metrics
    print("\nPer-class Metrics:")
    class_metrics = {}

    for i in range(num_classes):
        class_name = class_mapping[str(i)]
        true_positive = conf_matrix[i][i]
        false_positive = sum(conf_matrix[j][i] for j in range(num_classes) if j != i)
        false_negative = sum(conf_matrix[i][j] for j in range(num_classes) if j != i)
        true_negative = sum(conf_matrix[j][k] for j in range(num_classes) for k in range(num_classes)
                           if j != i and k != i)

        # Calculate metrics
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

        # Store metrics
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity
        }

        # Print metrics
        print(f"{class_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Specificity: {specificity:.4f}")

    # Calculate overall metrics
    overall_precision = sum(metrics['precision'] for metrics in class_metrics.values()) / num_classes
    overall_recall = sum(metrics['recall'] for metrics in class_metrics.values()) / num_classes
    overall_f1 = sum(metrics['f1_score'] for metrics in class_metrics.values()) / num_classes

    print("\nOverall Metrics:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1 Score: {overall_f1:.4f}")

    # Save metrics to file
    metrics_data = {
        'accuracy': test_acc,
        'confusion_matrix': conf_matrix.tolist(),
        'class_metrics': class_metrics,
        'overall_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1
        }
    }

    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=2)

    # Plot metrics
    plt.figure(figsize=(15, 10))

    # Plot confusion matrix
    plt.subplot(2, 2, 1)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [class_mapping[str(i)] for i in range(num_classes)], rotation=45)
    plt.yticks(tick_marks, [class_mapping[str(i)] for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Plot precision, recall, f1 by class
    plt.subplot(2, 2, 2)
    x = np.arange(num_classes)
    width = 0.2
    plt.bar(x - width, [class_metrics[class_mapping[str(i)]]['precision'] for i in range(num_classes)],
            width, label='Precision')
    plt.bar(x, [class_metrics[class_mapping[str(i)]]['recall'] for i in range(num_classes)],
            width, label='Recall')
    plt.bar(x + width, [class_metrics[class_mapping[str(i)]]['f1_score'] for i in range(num_classes)],
            width, label='F1 Score')
    plt.xticks(x, [class_mapping[str(i)] for i in range(num_classes)])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.title('Metrics by Class')

    # Plot overall metrics
    plt.subplot(2, 2, 3)
    overall_metrics = [test_acc, overall_precision, overall_recall, overall_f1]
    plt.bar(['Accuracy', 'Precision', 'Recall', 'F1 Score'], overall_metrics)
    plt.ylim(0, 1.1)
    plt.title('Overall Metrics')

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'metrics_plot.png'))
    print(f"Metrics plot saved to {os.path.join(model_dir, 'metrics_plot.png')}")

    return True

def generate_glasses(image, model_dir, output_dir):
    """Generate 3D glasses from an image"""
    if not TORCH_AVAILABLE or not TRIMESH_AVAILABLE:
        print("PyTorch or Trimesh not available. Cannot generate 3D glasses.")
        return None, None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save input image to temporary file if it's a PIL Image
    if isinstance(image, Image.Image):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name
    else:
        # Assume it's a path
        image_path = image
        image = Image.open(image_path).convert('RGB')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class mapping
    with open(os.path.join(model_dir, 'class_mapping.json'), 'r') as f:
        class_mapping = json.load(f)

    # Load model
    num_classes = len(class_mapping)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'eyeglasses_model.pth'), map_location=device))
    model.to(device)
    model.eval()

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Classify image
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    # Get class name and probabilities
    class_name = class_mapping[str(predicted_class)]
    probabilities = {class_mapping[str(i)]: float(probs[i]) for i in range(len(class_mapping))}

    print(f"Image classified as: {class_name} (confidence: {probabilities[class_name]:.2f})")

    # Generate 3D model based on classification
    has_glasses = class_name == 'with_glasses'

    # Create vertices for a simple glasses frame
    vertices = []
    faces = []

    # Parameters
    frame_width = 0.8 if has_glasses else 0.0  # No frame for no_glasses
    lens_size = 0.3 if has_glasses else 0.0    # No lens for no_glasses
    bridge_width = 0.2 if has_glasses else 0.0 # No bridge for no_glasses

    # Left lens (circle)
    left_center = [-frame_width/2, 0, 0]
    for i in range(10):
        angle = i * 2 * np.pi / 10
        vertices.append([left_center[0] + lens_size * np.cos(angle),
                         left_center[1] + lens_size * np.sin(angle),
                         left_center[2]])

    # Right lens (circle)
    right_center = [frame_width/2, 0, 0]
    for i in range(10):
        angle = i * 2 * np.pi / 10
        vertices.append([right_center[0] + lens_size * np.cos(angle),
                         right_center[1] + lens_size * np.sin(angle),
                         right_center[2]])

    # Bridge
    vertices.append([-bridge_width/2, 0, 0])  # 20
    vertices.append([bridge_width/2, 0, 0])   # 21

    # Temple arms
    vertices.append([-frame_width, 0, 0])     # 22
    vertices.append([frame_width, 0, 0])      # 23
    vertices.append([-frame_width*1.5, -0.5, 0])  # 24
    vertices.append([frame_width*1.5, -0.5, 0])   # 25

    # Create faces for left lens
    for i in range(9):
        faces.append([0, i+1, i+2])
    faces.append([0, 10, 1])

    # Create faces for right lens
    for i in range(9):
        faces.append([10, i+11, i+12])
    faces.append([10, 20, 11])

    # Create faces for bridge
    faces.append([20, 21, 0])
    faces.append([21, 10, 0])

    # Create faces for temple arms
    faces.append([0, 22, 24])
    faces.append([10, 23, 25])

    # Create a texture based on glasses type
    texture = np.zeros((256, 256, 3), dtype=np.uint8)

    if has_glasses:
        # Glasses texture (dark frame)
        for i in range(256):
            for j in range(256):
                if (i // 32 + j // 32) % 2 == 0:
                    texture[i, j] = [50, 50, 50]  # Dark gray
                else:
                    texture[i, j] = [30, 30, 30]  # Darker gray
    else:
        # No glasses texture (transparent)
        texture.fill(255)  # White/transparent

    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Save texture
    texture_path = os.path.join(output_dir, 'texture.png')
    Image.fromarray(texture).save(texture_path)

    # Save mesh
    output_path = os.path.join(output_dir, 'glasses.obj')
    mesh.export(output_path)

    # Create preview
    scene = trimesh.Scene(mesh)
    render = scene.save_image(resolution=[512, 512])
    preview = Image.open(io.BytesIO(render))
    preview_path = os.path.join(output_dir, 'preview.png')
    preview.save(preview_path)

    # Save classification results
    classification = {
        'class_name': class_name,
        'class_id': predicted_class,
        'probabilities': probabilities
    }

    with open(os.path.join(output_dir, 'classification.json'), 'w') as f:
        json.dump(classification, f, indent=2)

    # Create results visualization
    plt.figure(figsize=(12, 4))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    # Classification results
    plt.subplot(1, 3, 2)
    classes = list(probabilities.keys())
    probs = [probabilities[c] for c in classes]
    plt.barh(classes, probs)
    plt.title("Classification Results")
    plt.xlabel("Probability")

    # 3D model preview
    plt.subplot(1, 3, 3)
    plt.imshow(preview)
    plt.title("3D Model Preview")
    plt.axis('off')

    plt.tight_layout()
    results_path = os.path.join(output_dir, 'results.png')
    plt.savefig(results_path)

    return output_path, preview

def create_web_interface(model_dir, output_dir, data_dir=None):
    """Create a web interface for generating 3D glasses"""
    if not GRADIO_AVAILABLE:
        print("Gradio not available. Cannot create web interface.")
        return None

    # Use data_dir from args if not provided
    if data_dir is None:
        data_dir = "data/cv_eyeglasses"  # Default value

    # Load metrics if available
    metrics_path = os.path.join(model_dir, 'metrics.json')
    metrics_data = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")

    # Load class mapping
    class_mapping_path = os.path.join(model_dir, 'class_mapping.json')
    class_mapping = None
    if os.path.exists(class_mapping_path):
        try:
            with open(class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
        except Exception as e:
            print(f"Error loading class mapping: {e}")

    with gr.Blocks(title="3D Glasses Generator") as demo:
        gr.Markdown("# 3D Glasses Generator")
        gr.Markdown("Upload an image to generate a 3D glasses model")

        with gr.Tabs():
            with gr.TabItem("Generate 3D Glasses"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(type="pil", label="Input Image")
                        generate_btn = gr.Button("Generate 3D Glasses")

                    with gr.Column():
                        preview_image = gr.Image(label="3D Model Preview")
                        model_file = gr.File(label="Download 3D Model")

                        # Add classification results
                        classification_output = gr.JSON(label="Classification Results")

                # Add results visualization
                results_image = gr.Image(label="Results Visualization", visible=False)

                # Update function to show all outputs
                def generate_and_show(img):
                    model_path, preview = generate_glasses(img, model_dir, output_dir)

                    # Load classification results
                    classification = None
                    results_img = None

                    try:
                        with open(os.path.join(output_dir, 'classification.json'), 'r') as f:
                            classification = json.load(f)

                        results_img_path = os.path.join(output_dir, 'results.png')
                        if os.path.exists(results_img_path):
                            results_img = Image.open(results_img_path)
                    except Exception as e:
                        print(f"Error loading results: {e}")

                    # Update visibility of results image
                    results_image.visible = results_img is not None

                    return model_path, preview, classification, results_img

                generate_btn.click(
                    fn=generate_and_show,
                    inputs=[input_image],
                    outputs=[model_file, preview_image, classification_output, results_image]
                )

            # Add metrics tab if metrics are available
            if metrics_data:
                with gr.TabItem("Model Metrics"):
                    gr.Markdown("## Model Performance Metrics")

                    # Overall metrics
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Overall Metrics")
                            overall_metrics = metrics_data.get('overall_metrics', {})
                            accuracy = metrics_data.get('accuracy', 0)

                            metrics_md = f"""
                            - **Accuracy**: {accuracy:.4f}
                            - **Precision**: {overall_metrics.get('precision', 0):.4f}
                            - **Recall**: {overall_metrics.get('recall', 0):.4f}
                            - **F1 Score**: {overall_metrics.get('f1_score', 0):.4f}
                            """
                            gr.Markdown(metrics_md)

                        # Add metrics plot if available
                        metrics_plot_path = os.path.join(model_dir, 'metrics_plot.png')
                        if os.path.exists(metrics_plot_path):
                            with gr.Column():
                                gr.Image(metrics_plot_path, label="Metrics Visualization")

                    # Per-class metrics
                    gr.Markdown("### Per-Class Metrics")
                    class_metrics = metrics_data.get('class_metrics', {})

                    for class_name, metrics in class_metrics.items():
                        with gr.Accordion(f"Class: {class_name}", open=False):
                            class_md = f"""
                            - **Precision**: {metrics.get('precision', 0):.4f}
                            - **Recall**: {metrics.get('recall', 0):.4f}
                            - **F1 Score**: {metrics.get('f1_score', 0):.4f}
                            - **Specificity**: {metrics.get('specificity', 0):.4f}
                            """
                            gr.Markdown(class_md)

            # Add dataset info tab
            with gr.TabItem("Dataset Information"):
                gr.Markdown("## Dataset Information")

                if class_mapping:
                    classes_md = "### Classes\n"
                    for idx, name in class_mapping.items():
                        classes_md += f"- **Class {idx}**: {name}\n"

                    gr.Markdown(classes_md)

                # Add example images
                gr.Markdown("### Example Images")

                # Find example images
                example_images = []
                test_dir = os.path.join(data_dir, 'test')
                if os.path.exists(test_dir):
                    for ext in ['jpg', 'jpeg', 'png']:
                        example_images.extend(glob.glob(os.path.join(test_dir, f'*.{ext}')))

                    # Limit to 6 examples
                    if example_images:
                        if len(example_images) > 6:
                            example_images = example_images[:6]

                        with gr.Gallery(label="Example Images from Test Set").style(grid=3, height="auto"):
                            [Image.open(img_path) for img_path in example_images]

    return demo

def main():
    args = parse_args()

    # Check if running in Kaggle
    in_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
    if in_kaggle:
        print("Running in Kaggle environment")

    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'download' or args.mode == 'all':
        download_cv_dataset(args.data_dir)

    if args.mode == 'train' or args.mode == 'all':
        train_model(args.data_dir, args.model_dir, args.epochs, args.batch_size, args.no_cuda)

    if args.mode == 'web' or args.mode == 'all':
        demo = create_web_interface(args.model_dir, args.output_dir, args.data_dir)
        if demo:
            # In Kaggle, we need to use specific settings
            if in_kaggle:
                print("Launching web interface with Kaggle-specific settings...")
                demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
            else:
                # For local development
                demo.launch(share=True)

    if args.mode == 'generate' or args.mode == 'all':
        # Generate models for test images
        test_dir = os.path.join(args.data_dir, 'test')
        if os.path.exists(test_dir):
            test_images = []
            for ext in ['jpg', 'jpeg', 'png']:
                test_images.extend(glob.glob(os.path.join(test_dir, f'*.{ext}')))

            if test_images:
                # Limit to 3 images for demonstration
                if len(test_images) > 3:
                    test_images = test_images[:3]

                for image_path in test_images:
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    image_output_dir = os.path.join(args.output_dir, image_name)
                    os.makedirs(image_output_dir, exist_ok=True)

                    print(f"Generating model for {image_path}...")
                    generate_glasses(image_path, args.model_dir, image_output_dir)

if __name__ == "__main__":
    main()
