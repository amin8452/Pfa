#!/usr/bin/env python
"""
Evaluate the trained eyeglasses model and visualize results
"""

import os
import sys
import json
import argparse
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install PyTorch to evaluate the model.")

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate eyeglasses model')
    parser.add_argument('--data_dir', type=str, default='data/eyeglasses_dataset',
                        help='Directory containing the dataset')
    parser.add_argument('--model_dir', type=str, default='models/eyeglasses',
                        help='Directory containing the trained model')
    parser.add_argument('--output_dir', type=str, default='results/eyeglasses',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
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
        
        return image, label, image_path

def evaluate_model(args):
    """Evaluate the trained model on the test set"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot evaluate model.")
        return False
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define transform
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = EyeglassesDataset(
        os.path.join(args.data_dir, 'test_metadata.json'),
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # Load class mapping
    with open(os.path.join(args.data_dir, 'class_mapping.json'), 'r') as f:
        class_mapping = json.load(f)
    
    num_classes = len(class_mapping)
    class_names = [class_mapping[str(i)] for i in range(num_classes)]
    print(f"Evaluating with {num_classes} classes: {class_names}")
    
    # Load model
    model_path = os.path.join(args.model_dir, 'eyeglasses_model.pth')
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return False
    
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Evaluate model
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with open(os.path.join(args.output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Save detailed results
    results = []
    for i in range(len(all_labels)):
        results.append({
            'image_path': all_paths[i],
            'true_label': int(all_labels[i]),
            'true_class': class_names[all_labels[i]],
            'pred_label': int(all_preds[i]),
            'pred_class': class_names[all_preds[i]],
            'correct': all_labels[i] == all_preds[i],
            'probabilities': {class_names[j]: float(all_probs[i][j]) for j in range(num_classes)}
        })
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize some examples
    visualize_examples(results, args.output_dir, num_examples=5)
    
    return True

def visualize_examples(results, output_dir, num_examples=5):
    """Visualize some example predictions"""
    # Get correct and incorrect predictions
    correct = [r for r in results if r['correct']]
    incorrect = [r for r in results if not r['correct']]
    
    # Select random examples
    if len(correct) > 0:
        correct_examples = random.sample(correct, min(num_examples, len(correct)))
    else:
        correct_examples = []
    
    if len(incorrect) > 0:
        incorrect_examples = random.sample(incorrect, min(num_examples, len(incorrect)))
    else:
        incorrect_examples = []
    
    # Visualize correct examples
    if correct_examples:
        plt.figure(figsize=(15, 3 * len(correct_examples)))
        for i, example in enumerate(correct_examples):
            img = Image.open(example['image_path']).convert('RGB')
            
            plt.subplot(len(correct_examples), 2, i*2 + 1)
            plt.imshow(img)
            plt.title(f"True: {example['true_class']}")
            plt.axis('off')
            
            plt.subplot(len(correct_examples), 2, i*2 + 2)
            probs = [example['probabilities'][cls] for cls in example['probabilities']]
            classes = list(example['probabilities'].keys())
            plt.barh(classes, probs)
            plt.title(f"Pred: {example['pred_class']}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correct_examples.png'))
    
    # Visualize incorrect examples
    if incorrect_examples:
        plt.figure(figsize=(15, 3 * len(incorrect_examples)))
        for i, example in enumerate(incorrect_examples):
            img = Image.open(example['image_path']).convert('RGB')
            
            plt.subplot(len(incorrect_examples), 2, i*2 + 1)
            plt.imshow(img)
            plt.title(f"True: {example['true_class']}")
            plt.axis('off')
            
            plt.subplot(len(incorrect_examples), 2, i*2 + 2)
            probs = [example['probabilities'][cls] for cls in example['probabilities']]
            classes = list(example['probabilities'].keys())
            plt.barh(classes, probs)
            plt.title(f"Pred: {example['pred_class']}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'incorrect_examples.png'))

def main():
    args = parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.data_dir):
        print(f"Dataset directory {args.data_dir} not found.")
        print("Please run download_eyeglasses_dataset.py first.")
        return
    
    # Check if model exists
    if not os.path.exists(args.model_dir):
        print(f"Model directory {args.model_dir} not found.")
        print("Please run train_eyeglasses_model.py first.")
        return
    
    # Evaluate model
    if evaluate_model(args):
        print("Model evaluation completed successfully!")
    else:
        print("Model evaluation failed.")

if __name__ == "__main__":
    main()
