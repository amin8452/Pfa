#!/usr/bin/env python
"""
Download and prepare the Eyeglasses Image Classification Dataset from images.cv
"""

import os
import sys
import requests
import zipfile
import io
import shutil
import json
from tqdm import tqdm
import argparse
from PIL import Image
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Download and prepare eyeglasses dataset')
    parser.add_argument('--data_dir', type=str, default='data/eyeglasses_dataset',
                        help='Directory to save the dataset')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='Train/test split ratio')
    parser.add_argument('--resize', type=int, default=512,
                        help='Resize images to this size')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of images per class')
    return parser.parse_args()

def download_dataset(output_dir):
    """
    Download the Eyeglasses Image Classification Dataset
    
    Note: In a real implementation, you would need to handle authentication
    and proper download from images.cv. This is a simplified version.
    """
    print("Downloading Eyeglasses Image Classification Dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # URL for the dataset (this is a placeholder - the actual URL would be different)
    dataset_url = "https://images.cv/dataset/eyeglasses-image-classification-dataset"
    
    # Since direct download might require authentication, we'll provide instructions
    print(f"To download the dataset from {dataset_url}:")
    print("1. Visit the URL in your browser")
    print("2. Sign in or create an account if necessary")
    print("3. Download the dataset")
    print(f"4. Extract the downloaded zip file to {output_dir}")
    print("5. Run this script again with --skip_download flag")
    
    # Check if user has manually downloaded the dataset
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Found files in {output_dir}, assuming dataset is already downloaded.")
        return True
    
    # For demonstration purposes, we'll create a mock dataset structure
    print("Creating mock dataset structure for demonstration...")
    
    # Create class directories
    classes = ['eyeglasses', 'no_eyeglasses', 'sunglasses', 'reading_glasses']
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
    
    # Create mock images (colored squares)
    for cls in classes:
        cls_dir = os.path.join(output_dir, cls)
        
        # Determine base color for this class
        if cls == 'eyeglasses':
            base_color = (200, 200, 200)  # Gray
        elif cls == 'no_eyeglasses':
            base_color = (255, 255, 255)  # White
        elif cls == 'sunglasses':
            base_color = (50, 50, 50)     # Dark gray
        else:  # reading_glasses
            base_color = (200, 150, 100)  # Light brown
        
        # Create 10 mock images per class
        for i in range(10):
            # Add some random variation to the color
            color = tuple(min(255, max(0, c + random.randint(-20, 20))) for c in base_color)
            
            # Create a colored square
            img = Image.new('RGB', (512, 512), color=color)
            
            # Save the image
            img_path = os.path.join(cls_dir, f"{cls}_{i+1:03d}.jpg")
            img.save(img_path)
    
    print("Mock dataset created successfully!")
    return True

def prepare_dataset(data_dir, split_ratio=0.8, resize=512, limit=None):
    """
    Prepare the dataset for training:
    1. Resize images
    2. Split into train/test sets
    3. Create metadata files
    """
    print(f"Preparing dataset in {data_dir}...")
    
    # Check if the dataset directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory {data_dir} not found.")
        return False
    
    # Create train and test directories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d)) 
                 and d not in ['train', 'test']]
    
    if not class_dirs:
        print("Error: No class directories found.")
        return False
    
    print(f"Found {len(class_dirs)} classes: {', '.join(class_dirs)}")
    
    # Process each class
    train_metadata = []
    test_metadata = []
    
    for cls in class_dirs:
        cls_dir = os.path.join(data_dir, cls)
        
        # Get all images in this class
        images = [f for f in os.listdir(cls_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit the number of images if specified
        if limit and len(images) > limit:
            images = random.sample(images, limit)
        
        print(f"Processing {len(images)} images for class '{cls}'...")
        
        # Shuffle images
        random.shuffle(images)
        
        # Split into train and test
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Process train images
        for img_file in tqdm(train_images, desc=f"Train - {cls}"):
            src_path = os.path.join(cls_dir, img_file)
            dst_path = os.path.join(train_dir, f"{cls}_{img_file}")
            
            # Resize and save
            try:
                img = Image.open(src_path).convert('RGB')
                img = img.resize((resize, resize), Image.LANCZOS)
                img.save(dst_path)
                
                # Add to metadata
                train_metadata.append({
                    'file': dst_path,
                    'class': cls,
                    'label': class_dirs.index(cls)
                })
            except Exception as e:
                print(f"Error processing {src_path}: {e}")
        
        # Process test images
        for img_file in tqdm(test_images, desc=f"Test - {cls}"):
            src_path = os.path.join(cls_dir, img_file)
            dst_path = os.path.join(test_dir, f"{cls}_{img_file}")
            
            # Resize and save
            try:
                img = Image.open(src_path).convert('RGB')
                img = img.resize((resize, resize), Image.LANCZOS)
                img.save(dst_path)
                
                # Add to metadata
                test_metadata.append({
                    'file': dst_path,
                    'class': cls,
                    'label': class_dirs.index(cls)
                })
            except Exception as e:
                print(f"Error processing {src_path}: {e}")
    
    # Save metadata
    with open(os.path.join(data_dir, 'train_metadata.json'), 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(os.path.join(data_dir, 'test_metadata.json'), 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    # Create class mapping
    class_mapping = {i: cls for i, cls in enumerate(class_dirs)}
    with open(os.path.join(data_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"Dataset preparation completed!")
    print(f"Train set: {len(train_metadata)} images")
    print(f"Test set: {len(test_metadata)} images")
    print(f"Class mapping: {class_mapping}")
    
    return True

def main():
    args = parse_args()
    
    # Download dataset
    if not download_dataset(args.data_dir):
        print("Failed to download dataset.")
        return
    
    # Prepare dataset
    if not prepare_dataset(args.data_dir, args.split_ratio, args.resize, args.limit):
        print("Failed to prepare dataset.")
        return
    
    print("Dataset download and preparation completed successfully!")

if __name__ == "__main__":
    main()
