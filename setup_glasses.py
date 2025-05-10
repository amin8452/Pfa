import os
import argparse
import subprocess
import sys
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Setup Hunyuan3D Glasses Adaptation')
    parser.add_argument('--data_dir', type=str, default='data/glasses', help='Directory for glasses data')
    parser.add_argument('--download_hunyuan', action='store_true', help='Download Hunyuan3D-2 model')
    parser.add_argument('--create_dirs', action='store_true', help='Create necessary directories')
    parser.add_argument('--install_deps', action='store_true', help='Install dependencies')
    return parser.parse_args()

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False
    return True

def download_hunyuan():
    """Download Hunyuan3D-2 model using huggingface_hub"""
    try:
        from huggingface_hub import snapshot_download
        print("Downloading Hunyuan3D-2 model...")
        snapshot_download(repo_id="tencent/Hunyuan3D-2", local_dir="hunyuan_model")
        print("Download completed!")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("Trying pip install...")
        if run_command("pip install huggingface_hub"):
            print("huggingface_hub installed, please run the script again.")
        return False

def create_directories(data_dir):
    """Create necessary directories for the project"""
    dirs = [
        data_dir,
        f"{data_dir}/train/images",
        f"{data_dir}/train/meshes",
        f"{data_dir}/train/textures",
        f"{data_dir}/val/images",
        f"{data_dir}/val/meshes",
        f"{data_dir}/val/textures",
        f"{data_dir}/test/images",
        f"{data_dir}/test/meshes",
        f"{data_dir}/test/textures",
        "checkpoints",
        "results"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")
    
    # Create empty metadata files
    for split in ["train", "val", "test"]:
        metadata_file = f"{data_dir}/{split}_metadata.json"
        if not os.path.exists(metadata_file):
            with open(metadata_file, 'w') as f:
                f.write("[]")
            print(f"Created empty metadata file: {metadata_file}")

def install_dependencies():
    """Install required dependencies"""
    commands = [
        "pip install torch torchvision",
        "pip install transformers",
        "pip install trimesh",
        "pip install matplotlib",
        "pip install tqdm",
        "pip install pillow",
        "pip install numpy",
        "pip install huggingface_hub"
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            print(f"Failed to run: {cmd}")
            return False
    
    return True

def main():
    args = parse_args()
    
    if args.install_deps:
        print("Installing dependencies...")
        if not install_dependencies():
            print("Failed to install some dependencies.")
            return
    
    if args.download_hunyuan:
        if not download_hunyuan():
            print("Failed to download Hunyuan3D-2 model.")
    
    if args.create_dirs:
        print(f"Creating directories in {args.data_dir}...")
        create_directories(args.data_dir)
    
    print("\nSetup completed!")
    print("\nNext steps:")
    print("1. Prepare your glasses dataset according to the README")
    print("2. Train the model using train_glasses.py or finetune_glasses.py")
    print("3. Generate 3D glasses models using glasses_demo.py")

if __name__ == "__main__":
    main()
