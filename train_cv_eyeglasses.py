#!/usr/bin/env python
"""
Train and evaluate a model on the Eyeglasses Image Classification Dataset from images.cv
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
import time

# Try to import torch
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
    print("PyTorch not available. Please install PyTorch to train the model.")

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate model on eyeglasses dataset')
    parser.add_argument('--data_dir', type=str, default='data/cv_eyeglasses',
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='models/cv_eyeglasses',
                        help='Directory to save the trained model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate the model, no training')
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

def train_and_evaluate(args):
    """Train and evaluate a model on the eyeglasses dataset"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot train model.")
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
    
    # Define transforms
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = EyeglassesDataset(
        os.path.join(args.data_dir, 'train_metadata.json'),
        transform=data_transform
    )
    
    test_dataset = EyeglassesDataset(
        os.path.join(args.data_dir, 'test_metadata.json'),
        transform=data_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    # Load class mapping
    with open(os.path.join(args.data_dir, 'class_mapping.json'), 'r') as f:
        class_mapping = json.load(f)
    
    num_classes = len(class_mapping)
    print(f"Dataset has {num_classes} classes: {list(class_mapping.values())}")
    print(f"Training set: {len(train_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")
    
    # Create model
    model_path = os.path.join(args.output_dir, 'eyeglasses_model.pth')
    
    if args.eval_only and os.path.exists(model_path):
        # Load existing model for evaluation
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path} for evaluation")
    else:
        # Create new model for training
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded existing model from {model_path}")
            except Exception as e:
                print(f"Could not load existing model: {e}")
    
    model = model.to(device)
    
    # Training
    if not args.eval_only:
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Training loop
        train_losses = []
        train_accs = []
        
        start_time = time.time()
        
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training"):
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
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            print(f"Epoch {epoch+1}/{args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    
    # Evaluation
    print("\nEvaluating model on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    # Confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Update statistics
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Update confusion matrix
            for i in range(len(labels)):
                confusion_matrix[labels[i].item()][predicted[i].item()] += 1
    
    # Calculate accuracy
    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")
    
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
            row += f"{confusion_matrix[i][j]:15d}"
        print(row)
    
    # Calculate per-class metrics
    print("\nPer-class Metrics:")
    for i in range(num_classes):
        class_name = class_mapping[str(i)]
        true_positive = confusion_matrix[i][i]
        false_positive = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)
        false_negative = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    
    # Save evaluation results
    results = {
        'accuracy': test_acc,
        'confusion_matrix': confusion_matrix.tolist(),
        'class_mapping': class_mapping
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {os.path.join(args.output_dir, 'evaluation_results.json')}")
    
    return True

def main():
    args = parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.data_dir):
        print(f"Dataset directory {args.data_dir} not found.")
        print("Please run download_cv_eyeglasses.py first.")
        return
    
    # Train and evaluate model
    if train_and_evaluate(args):
        print("Model training and evaluation completed successfully!")
    else:
        print("Model training and evaluation failed.")

if __name__ == "__main__":
    main()
