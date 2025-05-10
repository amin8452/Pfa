#!/usr/bin/env python
"""
Hunyuan3D-Glasses Kaggle Setup
------------------------------
This script sets up the environment for running Hunyuan3D-Glasses in Kaggle.
It installs dependencies, downloads the model, and creates necessary directories.
"""

import os
import sys
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Setup Hunyuan3D-Glasses for Kaggle')
    parser.add_argument('--download_hunyuan', action='store_true', help='Download Hunyuan3D-2 model')
    parser.add_argument('--create_dirs', action='store_true', help='Create necessary directories')
    parser.add_argument('--install_deps', action='store_true', help='Install dependencies')
    parser.add_argument('--all', action='store_true', help='Do all setup steps')
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

def install_dependencies():
    """Install required dependencies for Kaggle"""
    print("\n=== Installing dependencies ===")
    
    # Check if running in Kaggle
    in_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
    if in_kaggle:
        print("Running in Kaggle environment")
    
    # List of required packages
    packages = [
        "torch",
        "torchvision",
        "transformers",
        "trimesh",
        "matplotlib",
        "tqdm",
        "pillow",
        "numpy",
        "huggingface_hub",
        "gradio",
        "mediapipe",
        "requests"
    ]
    
    # Install each package
    for package in packages:
        try:
            __import__(package.split('[')[0])  # Handle packages with extras like 'package[extra]'
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            if not run_command([sys.executable, "-m", "pip", "install", package]):
                print(f"Failed to install {package}. Please install it manually.")
    
    print("Dependencies installation completed!")

def download_hunyuan():
    """Download Hunyuan3D-2 model"""
    print("\n=== Downloading Hunyuan3D-2 model ===")
    
    if os.path.exists('hunyuan_model'):
        print("Hunyuan3D-2 model already exists. Skipping download.")
        return True
    
    try:
        from huggingface_hub import snapshot_download
        print("Downloading Hunyuan3D-2 model...")
        snapshot_download(repo_id="tencent/Hunyuan3D-2", local_dir="hunyuan_model")
        print("Download completed!")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("You can still use the custom model without Hunyuan3D.")
        return False

def create_directories():
    """Create necessary directories for Kaggle"""
    print("\n=== Creating directories ===")
    
    dirs = [
        "data/glasses/train/images",
        "data/glasses/val/images",
        "data/glasses/test/images",
        "results",
        "checkpoints"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")
    
    print("Directory creation completed!")

def download_sample_images():
    """Download some sample images for testing"""
    print("\n=== Downloading sample images ===")
    
    # Create examples directory
    os.makedirs("examples", exist_ok=True)
    
    # Sample image URLs
    sample_urls = {
        "eyeglasses1.jpg": "https://images.pexels.com/photos/701877/pexels-photo-701877.jpeg",
        "sunglasses1.jpg": "https://images.pexels.com/photos/46710/pexels-photo-46710.jpeg",
        "reading_glasses1.jpg": "https://images.pexels.com/photos/1438081/pexels-photo-1438081.jpeg",
        "sports_glasses1.jpg": "https://images.pexels.com/photos/1342609/pexels-photo-1342609.jpeg"
    }
    
    # Download each sample image
    import requests
    from PIL import Image
    
    for filename, url in sample_urls.items():
        output_path = os.path.join("examples", filename)
        
        if os.path.exists(output_path):
            print(f"Sample image {filename} already exists. Skipping download.")
            continue
        
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                img = Image.open(response.raw)
                img.save(output_path)
                print(f"Downloaded {filename}")
            else:
                print(f"Failed to download {filename}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    
    print("Sample images download completed!")

def main():
    args = parse_args()
    
    # If --all is specified, do everything
    if args.all:
        args.install_deps = True
        args.download_hunyuan = True
        args.create_dirs = True
    
    # Install dependencies
    if args.install_deps:
        install_dependencies()
    
    # Download Hunyuan3D-2 model
    if args.download_hunyuan:
        download_hunyuan()
    
    # Create directories
    if args.create_dirs:
        create_directories()
        download_sample_images()
    
    print("\n=== Setup completed! ===")
    print("You can now run the Hunyuan3D-Glasses pipeline.")

if __name__ == "__main__":
    main()
