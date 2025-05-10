#!/usr/bin/env python
"""
Hunyuan3D-Glasses Pipeline Runner
---------------------------------
This script runs the complete pipeline for generating 3D glasses models from 2D images.
It handles data collection, model setup, and 3D generation.
"""

import os
import sys
import argparse
import subprocess
import time
import webbrowser

def parse_args():
    parser = argparse.ArgumentParser(description='Run the Hunyuan3D-Glasses pipeline')
    parser.add_argument('--mode', choices=['setup', 'data', 'generate', 'web', 'all'], 
                        default='all', help='Mode to run')
    parser.add_argument('--image', type=str, default=None, 
                        help='Path to input image for direct generation')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Output directory')
    parser.add_argument('--data_dir', type=str, default='data/glasses', 
                        help='Data directory')
    parser.add_argument('--num_images', type=int, default=20, 
                        help='Number of images to download per category')
    parser.add_argument('--use_hunyuan', action='store_true', 
                        help='Use original Hunyuan3D model')
    parser.add_argument('--no_browser', action='store_true', 
                        help='Do not open browser automatically')
    return parser.parse_args()

def run_command(cmd, desc=None):
    """Run a command and print its output"""
    if desc:
        print(f"\n=== {desc} ===")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(f"Error: {result.stderr}")
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return False
    
    return True

def setup_environment():
    """Set up the environment for the pipeline"""
    print("\n=== Setting up environment ===")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Create necessary directories
    dirs = [
        'data/glasses/train/images',
        'data/glasses/val/images',
        'data/glasses/test/images',
        'results',
        'checkpoints'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")
    
    # Install dependencies
    requirements = [
        'torch',
        'torchvision',
        'transformers',
        'trimesh',
        'matplotlib',
        'tqdm',
        'pillow',
        'numpy',
        'huggingface_hub',
        'gradio',
        'mediapipe',
        'requests'
    ]
    
    for req in requirements:
        try:
            __import__(req)
            print(f"✓ {req} is already installed")
        except ImportError:
            print(f"Installing {req}...")
            if not run_command([sys.executable, '-m', 'pip', 'install', req]):
                print(f"Failed to install {req}. Please install it manually.")
    
    # Download Hunyuan3D model if needed
    if not os.path.exists('hunyuan_model'):
        try:
            from huggingface_hub import snapshot_download
            print("Downloading Hunyuan3D-2 model...")
            snapshot_download(repo_id="tencent/Hunyuan3D-2", local_dir="hunyuan_model")
            print("Download completed!")
        except Exception as e:
            print(f"Failed to download model: {e}")
            print("You can still use the custom model without Hunyuan3D.")
    
    print("Environment setup completed!")

def collect_data(data_dir, num_images):
    """Collect and prepare data for training"""
    print("\n=== Collecting and preparing data ===")
    
    # Run data collection script
    cmd = [
        sys.executable, 'src/data_collection.py',
        '--data_root', data_dir,
        '--scrape_images',
        '--process_images',
        '--augment_data',
        '--split_data',
        '--num_images', str(num_images)
    ]
    
    if not run_command(cmd, "Downloading and processing images"):
        print("Data collection failed. Please check the error messages.")
        return False
    
    print("Data collection completed!")
    return True

def generate_glasses(image_path, output_dir, use_hunyuan=False):
    """Generate 3D glasses from an image"""
    print(f"\n=== Generating 3D glasses from {image_path} ===")
    
    # Run glasses demo script
    cmd = [
        sys.executable, 'src/glasses_demo.py',
        '--image', image_path,
        '--output_dir', output_dir
    ]
    
    if use_hunyuan:
        cmd.append('--use_hunyuan')
    
    if not run_command(cmd, "Generating 3D glasses"):
        print("Generation failed. Please check the error messages.")
        return False
    
    print(f"Generation completed! Results saved to {output_dir}")
    return True

def run_web_interface():
    """Run the web interface"""
    print("\n=== Starting web interface ===")
    
    # Start the web interface in a separate process
    cmd = [sys.executable, 'glasses_app.py']
    
    # Use Popen to run in background
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait a bit for the server to start
    time.sleep(5)
    
    # Check if process is still running
    if process.poll() is None:
        print("Web interface started successfully!")
        
        # Open browser
        webbrowser.open('http://localhost:7860')
        
        print("Press Ctrl+C to stop the web interface")
        
        try:
            # Print output from the process
            while True:
                output = process.stdout.readline()
                if output:
                    print(output.strip())
                
                # Check if process is still running
                if process.poll() is not None:
                    break
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping web interface...")
            process.terminate()
    else:
        stdout, stderr = process.communicate()
        print(f"Web interface failed to start: {stderr}")
        return False
    
    return True

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the selected mode
    if args.mode == 'setup' or args.mode == 'all':
        setup_environment()
    
    if args.mode == 'data' or args.mode == 'all':
        collect_data(args.data_dir, args.num_images)
    
    if args.mode == 'generate' or args.image:
        # If image is provided, use it directly
        if args.image:
            generate_glasses(args.image, args.output_dir, args.use_hunyuan)
        # Otherwise, use a sample image from the dataset
        else:
            # Find a sample image
            sample_dir = os.path.join(args.data_dir, 'test', 'images')
            if os.path.exists(sample_dir) and os.listdir(sample_dir):
                sample_image = os.path.join(sample_dir, os.listdir(sample_dir)[0])
                generate_glasses(sample_image, args.output_dir, args.use_hunyuan)
            else:
                print("No sample images found. Please run data collection first or provide an image.")
    
    if args.mode == 'web' or args.mode == 'all':
        run_web_interface()

if __name__ == "__main__":
    main()
