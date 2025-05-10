import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from tqdm import tqdm
import wandb  # Pour le suivi des expériences

from model_glasses import GlassesHunyuan3DModel
from data_loader_glasses import GlassesDataset
from metrics import chamfer_distance, f_score, normal_consistency, texture_similarity
from utils import save_mesh, visualize_reconstruction

# Configuration
config = {
    'batch_size': 4,
    'epochs': 100,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'checkpoint_dir': 'checkpoints',
    'log_interval': 10,
    'eval_interval': 5,
    'save_interval': 10,
    'hunyuan_checkpoint': 'hy3dgen/checkpoints/hunyuan3d_v2.pth',
    'data_root': 'data/glasses',
    'img_size': 256,
    'use_wandb': True,
    'wandb_project': 'glasses-3d-reconstruction'
}

# Initialiser wandb
if config['use_wandb']:
    wandb.init(project=config['wandb_project'], config=config)

# Créer les répertoires nécessaires
os.makedirs(config['checkpoint_dir'], exist_ok=True)

# Définir le device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Charger les datasets
train_dataset = GlassesDataset(root_dir=config['data_root'], mode='train', img_size=config['img_size'])
val_dataset = GlassesDataset(root_dir=config['data_root'], mode='val', img_size=config['img_size'], augment=False)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Initialiser le modèle
model = GlassesHunyuan3DModel(checkpoint_path=config['hunyuan_checkpoint'])
model.to(device)

# Définir l'optimiseur
optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Fonction de perte
def compute_loss(pred, target):
    losses = {}
    
    # Chamfer Distance
    losses['chamfer'] = chamfer_distance(pred['vertices'], target['vertices'])
    
    # Normal consistency si les faces sont disponibles
    if 'faces' in pred and 'faces' in target:
        losses['normal'] = 0.1 * (1.0 - normal_consistency(
            pred['vertices'], pred['faces'],
            target['vertices'], target['faces']
        ))
    else:
        losses['normal'] = torch.tensor(0.0, device=device)
    
    # Texture loss si les textures sont disponibles
    if 'texture' in pred and 'texture' in target:
        losses['texture'] = 0.05 * (1.0 - texture_similarity(pred['texture'], target['texture']))
    else:
        losses['texture'] = torch.tensor(0.0, device=device)
    
    # Perte totale
    losses['total'] = losses['chamfer'] + losses['normal'] + losses['texture']
    
    return losses

# Fonction d'évaluation
def evaluate(model, data_loader):
    model.eval()
    total_losses = {}
    total_metrics = {'chamfer': 0.0, 'f_score': 0.0, 'normal_consistency': 0.0, 'texture_similarity': 0.0}
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Préparer les données
            images = batch['image'].to(device)
            target = {
                'vertices': batch['vertices'].to(device),
                'faces': batch['faces'].to(device) if 'faces' in batch else None,
                'texture': batch['texture'].to(device) if 'texture' in batch else None
            }
            
            # Forward pass
            pred = model(images)
            
            # Calculer les pertes
            losses = compute_loss(pred, target)
            
            # Calculer les métriques
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
            
            # Accumuler les pertes et métriques
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            
            for k, v in metrics.items():
                total_metrics[k] += v
            
            count += 1
    
    # Calculer les moyennes
    for k in total_losses:
        total_losses[k] /= count
    
    for k in total_metrics:
        total_metrics[k] /= count
    
    return total_losses, total_metrics

# Fonction d'entraînement
def train_epoch(model, data_loader, optimizer, epoch):
    model.train()
    total_losses = {}
    count = 0
    
    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # Préparer les données
        images = batch['image'].to(device)
        target = {
            'vertices': batch['vertices'].to(device),
            'faces': batch['faces'].to(device) if 'faces' in batch else None,
            'texture': batch['texture'].to(device) if 'texture' in batch else None
        }
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(images)
        
        # Calculer les pertes
        losses = compute_loss(pred, target)
        
        # Backward pass
        losses['total'].backward()
        optimizer.step()
        
        # Accumuler les pertes
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        
        count += 1
        
        # Afficher les pertes
        if batch_idx % config['log_interval'] == 0:
            log_str = f"Train Epoch: {epoch} [{batch_idx}/{len(data_loader)}] "
            for k, v in losses.items():
                log_str += f"{k}: {v.item():.4f} "
            print(log_str)
            
            if config['use_wandb']:
                wandb.log({f"train/{k}": v.item() for k, v in losses.items()})
    
    # Calculer les moyennes
    for k in total_losses:
        total_losses[k] /= count
    
    return total_losses

# Boucle d'entraînement principale
best_val_loss = float('inf')

for epoch in range(1, config['epochs'] + 1):
    # Entraînement
    train_losses = train_epoch(model, train_loader, optimizer, epoch)
    
    # Log des pertes d'entraînement
    print(f"Epoch {epoch} Train Losses:")
    for k, v in train_losses.items():
        print(f"  {k}: {v:.4f}")
    
    if config['use_wandb']:
        wandb.log({f"train_epoch/{k}": v for k, v in train_losses.items()})
    
    # Évaluation
    if epoch % config['eval_interval'] == 0:
        val_losses, val_metrics = evaluate(model, val_loader)
        
        # Log des pertes et métriques de validation
        print(f"Epoch {epoch} Validation Losses:")
        for k, v in val_losses.items():
            print(f"  {k}: {v:.4f}")
        
        print(f"Epoch {epoch} Validation Metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        if config['use_wandb']:
            wandb.log({f"val/{k}": v for k, v in val_losses.items()})
            wandb.log({f"metrics/{k}": v for k, v in val_metrics.items()})
        
        # Mettre à jour le scheduler
        scheduler.step(val_losses['total'])
        
        # Sauvegarder le meilleur modèle
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'val_metrics': val_metrics
            }, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    # Sauvegarder le modèle périodiquement
    if epoch % config['save_interval'] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(config['checkpoint_dir'], f'model_epoch_{epoch}.pth'))

print("Training completed!")