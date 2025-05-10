#!/usr/bin/env python
"""
Download and prepare the Eyeglasses Image Classification Dataset from images.cv
This script downloads and prepares the dataset for training the Hunyuan3D-Glasses model.
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
    parser = argparse.ArgumentParser(description='Download and prepare eyeglasses dataset from images.cv')
    parser.add_argument('--data_dir', type=str, default='data/cv_eyeglasses',
                        help='Directory to save the dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train/test split ratio')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Size to resize images')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key for images.cv (if needed)')
    return parser.parse_args()

def download_cv_dataset(output_dir, api_key=None):
    """
    Download the Eyeglasses Image Classification Dataset from images.cv
    
    Args:
        output_dir: Directory to save the dataset
        api_key: API key for images.cv (if needed)
    """
    print("Downloading Eyeglasses Image Classification Dataset from images.cv...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset URL
    dataset_url = "https://images.cv/dataset/eyeglasses-image-classification-dataset"
    
    # Print instructions for manual download
    print(f"To download the dataset from {dataset_url}:")
    print("1. Visit the URL in your browser")
    print("2. Sign in or create an account if necessary")
    print("3. Download the dataset")
    print(f"4. Extract the downloaded zip file to {output_dir}")
    
    # Check if user has manually downloaded the dataset
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Found files in {output_dir}, assuming dataset is already downloaded.")
        return True
    
    # Try to download automatically if API key is provided
    if api_key:
        try:
            # This is a placeholder - the actual API call would depend on images.cv's API
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(f"{dataset_url}/download", headers=headers)
            
            if response.status_code == 200:
                # Extract the zip file
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(output_dir)
                print(f"Dataset downloaded and extracted to {output_dir}")
                return True
            else:
                print(f"Failed to download dataset: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
    
    return False

def prepare_dataset(data_dir, train_ratio=0.8, image_size=512):
    """
    Prepare the dataset for training:
    1. Organize files into a standard structure
    2. Resize images
    3. Split into train/test sets
    4. Create metadata files
    """
    print(f"Preparing dataset in {data_dir}...")
    
    # Expected structure of the images.cv dataset
    # - with_glasses/
    # - without_glasses/
    
    # Create train and test directories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Define class mapping
    class_mapping = {
        0: "with_glasses",
        1: "without_glasses"
    }
    
    # Save class mapping
    with open(os.path.join(data_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Process each class
    train_metadata = []
    test_metadata = []
    
    for class_idx, class_name in class_mapping.items():
        class_dir = os.path.join(data_dir, class_name)
        
        # Skip if directory doesn't exist
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory {class_dir} not found.")
            continue
        
        # Get all images in this class
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        print(f"Processing {len(images)} images for class '{class_name}'...")
        
        # Shuffle images
        random.shuffle(images)
        
        # Split into train and test
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Process train images
        for img_file in tqdm(train_images, desc=f"Train - {class_name}"):
            src_path = os.path.join(class_dir, img_file)
            dst_path = os.path.join(train_dir, f"{class_name}_{img_file}")
            
            # Resize and save
            try:
                img = Image.open(src_path).convert('RGB')
                img = img.resize((image_size, image_size), Image.LANCZOS)
                img.save(dst_path)
                
                # Add to metadata
                train_metadata.append({
                    'file': dst_path,
                    'class': class_name,
                    'label': class_idx
                })
            except Exception as e:
                print(f"Error processing {src_path}: {e}")
        
        # Process test images
        for img_file in tqdm(test_images, desc=f"Test - {class_name}"):
            src_path = os.path.join(class_dir, img_file)
            dst_path = os.path.join(test_dir, f"{class_name}_{img_file}")
            
            # Resize and save
            try:
                img = Image.open(src_path).convert('RGB')
                img = img.resize((image_size, image_size), Image.LANCZOS)
                img.save(dst_path)
                
                # Add to metadata
                test_metadata.append({
                    'file': dst_path,
                    'class': class_name,
                    'label': class_idx
                })
            except Exception as e:
                print(f"Error processing {src_path}: {e}")
    
    # Save metadata
    with open(os.path.join(data_dir, 'train_metadata.json'), 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(os.path.join(data_dir, 'test_metadata.json'), 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"Dataset preparation completed!")
    print(f"Train set: {len(train_metadata)} images")
    print(f"Test set: {len(test_metadata)} images")
    
    return True

def main():
    args = parse_args()
    
    # Download dataset
    download_cv_dataset(args.data_dir, args.api_key)
    
    # Prepare dataset
    prepare_dataset(args.data_dir, args.train_ratio, args.image_size)
    
    print("Dataset download and preparation completed!")
    print(f"Dataset saved to {args.data_dir}")
    print("You can now train the model using:")
    print(f"python train_eyeglasses_model.py --data_dir {args.data_dir}")

if __name__ == "__main__":
    main()
