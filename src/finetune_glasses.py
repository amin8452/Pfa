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
from metrics import chamfer_distance, f_score, normal_consistency, texture_similarity

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune GlassesHunyuan3DModel')
    parser.add_argument('--data_root', type=str, default='data/glasses', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to model checkpoint or HuggingFace model ID')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze the image encoder')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=10, help='Checkpoint saving interval')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    return parser.parse_args()

def compute_loss(pred, target, device):
    """
    Compute the loss between predicted and target 3D models.
    
    Args:
        pred: Dictionary containing predicted vertices, faces, and texture
        target: Dictionary containing target vertices, faces, and texture
        device: Device to compute the loss on
        
    Returns:
        Dictionary of losses
    """
    losses = {}
    
    # Chamfer Distance
    losses['chamfer'] = chamfer_distance(pred['vertices'], target['vertices'])
    
    # Normal consistency if faces are available
    if 'faces' in pred and 'faces' in target:
        losses['normal'] = 0.1 * (1.0 - normal_consistency(
            pred['vertices'], pred['faces'],
            target['vertices'], target['faces']
        ))
    else:
        losses['normal'] = torch.tensor(0.0, device=device)
    
    # Texture loss if textures are available
    if 'texture' in pred and 'texture' in target:
        losses['texture'] = 0.05 * (1.0 - texture_similarity(pred['texture'], target['texture']))
    else:
        losses['texture'] = torch.tensor(0.0, device=device)
    
    # Total loss
    losses['total'] = losses['chamfer'] + losses['normal'] + losses['texture']
    
    return losses

def evaluate(model, data_loader, device):
    """
    Evaluate the model on the validation set.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the validation set
        device: Device to compute on
        
    Returns:
        Tuple of (losses, metrics)
    """
    model.eval()
    total_losses = {}
    total_metrics = {'chamfer': 0.0, 'f_score': 0.0, 'normal_consistency': 0.0, 'texture_similarity': 0.0}
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Prepare data
            images = batch['image'].to(device)
            target = {
                'vertices': batch['vertices'].to(device),
                'faces': batch['faces'].to(device) if 'faces' in batch else None,
                'texture': batch['texture'].to(device) if 'texture' in batch else None
            }
            
            # Forward pass
            pred = model(images)
            
            # Compute losses
            losses = compute_loss(pred, target, device)
            
            # Compute metrics
            metrics = {
                'chamfer': chamfer_distance(pred['vertices'], target['vertices']).item(),
                'f_score': f_score(pred['vertices'], target['vertices']).item()
            }
            
            if 'faces' in pred and 'faces' in target:
                metrics['normal_consistency'] = normal_consistency(
                    pred['vertices'], pred['faces'],
                    target['vertices'], target['faces']
                ).item()
            
            if 'texture' in pred and 'texture' in target:
                metrics['texture_similarity'] = texture_similarity(
                    pred['texture'], target['texture']
                ).item()
            
            # Accumulate losses and metrics
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            
            for k, v in metrics.items():
                total_metrics[k] += v
            
            count += 1
    
    # Compute averages
    for k in total_losses:
        total_losses[k] /= count
    
    for k in total_metrics:
        total_metrics[k] /= count
    
    return total_losses, total_metrics

def train_epoch(model, data_loader, optimizer, device, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        data_loader: DataLoader for the training set
        optimizer: Optimizer
        device: Device to compute on
        epoch: Current epoch number
        
    Returns:
        Dictionary of average losses
    """
    model.train()
    total_losses = {}
    count = 0
    
    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # Prepare data
        images = batch['image'].to(device)
        target = {
            'vertices': batch['vertices'].to(device),
            'faces': batch['faces'].to(device) if 'faces' in batch else None,
            'texture': batch['texture'].to(device) if 'texture' in batch else None
        }
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(images)
        
        # Compute losses
        losses = compute_loss(pred, target, device)
        
        # Backward pass
        losses['total'].backward()
        optimizer.step()
        
        # Accumulate losses
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        
        count += 1
    
    # Compute averages
    for k in total_losses:
        total_losses[k] /= count
    
    return total_losses

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        import wandb
        wandb.init(project="glasses-3d-reconstruction", config=vars(args))
    
    # Load datasets
    train_dataset = GlassesDataset(root_dir=args.data_root, mode='train', img_size=args.img_size)
    val_dataset = GlassesDataset(root_dir=args.data_root, mode='val', img_size=args.img_size, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model
    model = GlassesHunyuan3DModel(checkpoint_path=args.checkpoint, freeze_encoder=args.freeze_encoder)
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Log training losses
        print(f"Epoch {epoch} Train Losses:")
        for k, v in train_losses.items():
            print(f"  {k}: {v:.4f}")
        
        if args.use_wandb:
            wandb.log({f"train/{k}": v for k, v in train_losses.items()})
        
        # Evaluate
        if epoch % args.eval_interval == 0:
            val_losses, val_metrics = evaluate(model, val_loader, device)
            
            # Log validation losses and metrics
            print(f"Epoch {epoch} Validation Losses:")
            for k, v in val_losses.items():
                print(f"  {k}: {v:.4f}")
            
            print(f"Epoch {epoch} Validation Metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            if args.use_wandb:
                wandb.log({f"val/{k}": v for k, v in val_losses.items()})
                wandb.log({f"metrics/{k}": v for k, v in val_metrics.items()})
            
            # Update scheduler
            scheduler.step(val_losses['total'])
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_losses['total'],
                    'val_metrics': val_metrics
                }, os.path.join(args.output_dir, 'best_model.pth'))
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'))
    
    print("Training completed!")

if __name__ == '__main__':
    main()
