#!/usr/bin/env python
"""
Run the complete pipeline for generating 3D glasses models using the images.cv dataset
"""

import os
import sys
import argparse
import subprocess
import time
import glob
from PIL import Image
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Run the complete glasses pipeline with images.cv dataset')
    parser.add_argument('--mode', choices=['download', 'train', 'generate', 'all'], default='all',
                        help='Mode to run')
    parser.add_argument('--data_dir', type=str, default='data/cv_eyeglasses',
                        help='Directory for the dataset')
    parser.add_argument('--model_dir', type=str, default='models/cv_eyeglasses',
                        help='Directory for the trained model')
    parser.add_argument('--output_dir', type=str, default='results/cv_glasses',
                        help='Directory for the generated models')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to input image for generation (if not specified, will use test images)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    return parser.parse_args()

def run_command(cmd, desc=None):
    """Run a command and print its output"""
    if desc:
        print(f"\n=== {desc} ===")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Get return code
        return_code = process.poll()
        
        # Print any errors
        if return_code != 0:
            print("Command failed with return code", return_code)
            for line in process.stderr.readlines():
                print(line.strip())
            return False
        
        return True
    
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def download_dataset(args):
    """Download and prepare the dataset"""
    print("\n=== Downloading and preparing dataset ===")
    
    cmd = [
        sys.executable, 'download_cv_eyeglasses.py',
        '--data_dir', args.data_dir
    ]
    
    return run_command(cmd, "Downloading dataset")

def train_model(args):
    """Train the model on the dataset"""
    print("\n=== Training model ===")
    
    cmd = [
        sys.executable, 'train_cv_eyeglasses.py',
        '--data_dir', args.data_dir,
        '--output_dir', args.model_dir,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size)
    ]
    
    if args.no_cuda:
        cmd.append('--no_cuda')
    
    return run_command(cmd, "Training model")

def generate_models(args):
    """Generate 3D models from images"""
    print("\n=== Generating 3D models ===")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If image path is specified, generate model for that image
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: Image file not found: {args.image_path}")
            return False
        
        cmd = [
            sys.executable, 'generate_3d_glasses.py',
            '--image_path', args.image_path,
            '--model_dir', args.model_dir,
            '--output_dir', args.output_dir
        ]
        
        if args.no_cuda:
            cmd.append('--no_cuda')
        
        return run_command(cmd, f"Generating model for {args.image_path}")
    
    # Otherwise, generate models for test images
    test_dir = os.path.join(args.data_dir, 'test')
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        return False
    
    # Find test images
    test_images = []
    for ext in ['jpg', 'jpeg', 'png']:
        test_images.extend(glob.glob(os.path.join(test_dir, f'*.{ext}')))
    
    if not test_images:
        print(f"Error: No test images found in {test_dir}")
        return False
    
    # Limit to 5 images for demonstration
    if len(test_images) > 5:
        test_images = test_images[:5]
    
    # Generate models for each test image
    success = True
    for i, image_path in enumerate(test_images):
        print(f"\nGenerating model for test image {i+1}/{len(test_images)}: {image_path}")
        
        # Create output directory for this image
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(args.output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        cmd = [
            sys.executable, 'generate_3d_glasses.py',
            '--image_path', image_path,
            '--model_dir', args.model_dir,
            '--output_dir', image_output_dir
        ]
        
        if args.no_cuda:
            cmd.append('--no_cuda')
        
        if not run_command(cmd, f"Generating model for {image_path}"):
            success = False
    
    # Create summary of results
    if success:
        create_results_summary(args.output_dir, test_images)
    
    return success

def create_results_summary(output_dir, test_images):
    """Create a summary of the generation results"""
    print("\n=== Creating results summary ===")
    
    # Create figure
    num_images = len(test_images)
    plt.figure(figsize=(15, 5 * num_images))
    
    for i, image_path in enumerate(test_images):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, image_name)
        
        # Original image
        plt.subplot(num_images, 3, i*3 + 1)
        img = Image.open(image_path).convert('RGB')
        plt.imshow(img)
        plt.title(f"Input: {image_name}")
        plt.axis('off')
        
        # 3D model preview
        preview_path = os.path.join(image_output_dir, 'preview.png')
        if os.path.exists(preview_path):
            plt.subplot(num_images, 3, i*3 + 2)
            preview = Image.open(preview_path)
            plt.imshow(preview)
            plt.title(f"3D Model: {image_name}")
            plt.axis('off')
        
        # Results visualization
        results_path = os.path.join(image_output_dir, 'results.png')
        if os.path.exists(results_path):
            plt.subplot(num_images, 3, i*3 + 3)
            results = Image.open(results_path)
            plt.imshow(results)
            plt.title(f"Results: {image_name}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary.png'))
    print(f"Results summary saved to {os.path.join(output_dir, 'summary.png')}")

def main():
    args = parse_args()
    
    start_time = time.time()
    
    if args.mode == 'download' or args.mode == 'all':
        if not download_dataset(args):
            print("Dataset download failed.")
            return
    
    if args.mode == 'train' or args.mode == 'all':
        if not train_model(args):
            print("Model training failed.")
            return
    
    if args.mode == 'generate' or args.mode == 'all':
        if not generate_models(args):
            print("Model generation failed.")
            return
    
    total_time = time.time() - start_time
    print(f"\n=== Pipeline completed in {total_time:.2f} seconds ===")
    
    # Print summary
    print("\nSummary:")
    print(f"- Dataset: {args.data_dir}")
    print(f"- Model: {args.model_dir}")
    print(f"- Results: {args.output_dir}")
    
    if os.path.exists(os.path.join(args.output_dir, 'summary.png')):
        print(f"- Results summary: {os.path.join(args.output_dir, 'summary.png')}")
    
    print("\nYou can now use the trained model to generate 3D glasses models from new images:")
    print(f"python generate_3d_glasses.py --image_path <path_to_image> --model_dir {args.model_dir} --output_dir <output_directory>")

if __name__ == "__main__":
    main()
