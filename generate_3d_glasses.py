#!/usr/bin/env python
"""
Generate 3D glasses models using the trained eyeglasses classifier
"""

import os
import sys
import json
import argparse
import numpy as np
import random
from PIL import Image
import trimesh
import matplotlib.pyplot as plt

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using mock implementation.")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate 3D glasses models')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model_dir', type=str, default='models/cv_eyeglasses',
                        help='Directory containing the trained model')
    parser.add_argument('--output_dir', type=str, default='results/3d_glasses',
                        help='Directory to save generated models')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    return parser.parse_args()

def classify_image(image_path, model_path, class_mapping, device):
    """Classify an image using the trained model"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot classify image.")
        return None
    
    # Load model
    num_classes = len(class_mapping)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Classify image
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
    
    # Get class name and probabilities
    class_name = class_mapping[str(predicted_class)]
    probabilities = {class_mapping[str(i)]: float(probs[i]) for i in range(len(class_mapping))}
    
    return {
        'class_name': class_name,
        'class_id': predicted_class,
        'probabilities': probabilities
    }

def generate_simple_glasses_model(has_glasses=True):
    """Generate a simple glasses model"""
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
    
    return mesh, texture

def generate_3d_glasses(image_path, model_dir, output_dir, no_cuda=False):
    """Generate a 3D glasses model based on image classification"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    use_cuda = torch.cuda.is_available() and not no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load class mapping
    class_mapping_path = os.path.join(model_dir, 'class_mapping.json')
    if not os.path.exists(class_mapping_path):
        print(f"Class mapping file not found at {class_mapping_path}")
        return False
    
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Load model
    model_path = os.path.join(model_dir, 'eyeglasses_model.pth')
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return False
    
    # Classify image
    print(f"Classifying image: {image_path}")
    classification = classify_image(image_path, model_path, class_mapping, device)
    
    if classification is None:
        print("Image classification failed.")
        return False
    
    print(f"Image classified as: {classification['class_name']} (confidence: {classification['probabilities'][classification['class_name']]:.2f})")
    print("Class probabilities:")
    for class_name, prob in classification['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
    
    # Generate 3D model based on classification
    print("Generating 3D model...")
    has_glasses = classification['class_name'] == 'with_glasses'
    mesh, texture = generate_simple_glasses_model(has_glasses)
    
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
    with open(os.path.join(output_dir, 'classification.json'), 'w') as f:
        json.dump(classification, f, indent=2)
    
    print(f"3D model generated and saved to {output_path}")
    print(f"Preview saved to {preview_path}")
    
    # Display results
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    img = Image.open(image_path).convert('RGB')
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')
    
    # Classification results
    plt.subplot(1, 3, 2)
    classes = list(classification['probabilities'].keys())
    probs = [classification['probabilities'][c] for c in classes]
    plt.barh(classes, probs)
    plt.title("Classification Results")
    plt.xlabel("Probability")
    
    # 3D model preview
    plt.subplot(1, 3, 3)
    preview = Image.open(preview_path)
    plt.imshow(preview)
    plt.title("3D Model Preview")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results.png'))
    
    return True

def main():
    args = parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Image file not found: {args.image_path}")
        return
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Model directory not found: {args.model_dir}")
        print("Please train the model first using train_cv_eyeglasses.py")
        return
    
    # Generate 3D glasses
    if generate_3d_glasses(args.image_path, args.model_dir, args.output_dir, args.no_cuda):
        print("3D glasses generation completed successfully!")
    else:
        print("3D glasses generation failed.")

if __name__ == "__main__":
    main()
