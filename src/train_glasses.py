# Training script for glasses 3D reconstruction
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from model_glasses import GlassesHunyuan3DModel
from data_loader_glasses import GlassesDataset
from model import Simple3DModel
from metrics import chamfer_distance

def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D glasses reconstruction model')
    parser.add_argument('--data_root', type=str, default='data/glasses', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--use_hunyuan', action='store_true', help='Use GlassesHunyuan3DModel instead of Simple3DModel')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to Hunyuan3D checkpoint')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    train_dataset = GlassesDataset(root_dir=args.data_root, mode='train', img_size=args.img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print(f"Train dataset size: {len(train_dataset)}")

    # Initialize model
    if args.use_hunyuan:
        print("Using GlassesHunyuan3DModel")
        model = GlassesHunyuan3DModel(checkpoint_path=args.checkpoint)
    else:
        print("Using Simple3DModel")
        model = Simple3DModel()

    model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # Prepare data
            images = batch['image'].to(device)
            target_vertices = batch['vertices'].to(device)

            # Forward pass
            optimizer.zero_grad()

            if args.use_hunyuan:
                output = model(images)
                pred_vertices = output['vertices']
            else:
                pred_vertices = model(images)

            # Compute loss
            loss = chamfer_distance(pred_vertices, target_vertices)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Compute average loss
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'))

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(args.output_dir, 'model_final.pth'))

    print("Training completed!")

if __name__ == '__main__':
    main()
