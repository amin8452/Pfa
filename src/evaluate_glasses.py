import torch
import os
import json
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model_glasses import GlassesHunyuan3DModel
from data_loader_glasses import GlassesDataset
from metrics import evaluate_glasses_reconstruction
from utils import save_mesh, visualize_reconstruction

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 3D glasses reconstruction model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='data/glasses', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Définir le device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Charger le dataset de test
    test_dataset = GlassesDataset(root_dir=args.data_root, mode='test', img_size=args.img_size, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Initialiser le modèle
    model = GlassesHunyuan3DModel()
    
    # Charger les poids du modèle
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Évaluer le modèle
    all_metrics = []
    visualized_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Préparer les données
            images = batch['image'].to(device)
            item_ids = batch['item_id']
            
            target = {
                'vertices': batch['vertices'].to(device),
                'faces': batch['faces'].to(device) if 'faces' in batch else None,
                'texture': batch['texture'].to(device) if 'texture' in batch else None
            }
            
            # Forward pass
            pred = model(images)
            
            # Calculer les métriques
            batch_metrics = []
            for i in range(images.size(0)):
                pred_sample = {k: v[i:i+1] for k, v in pred.items()}
                target_sample = {k: v[i:i+1] for k, v in target.items()}
                
                metrics = evaluate_glasses_reconstruction(pred_sample, target_sample)
                metrics['item_id'] = item_ids[i]
                batch_metrics.append(metrics)
                
                # Visualiser quelques échantillons
                if visualized_samples < args.num_samples:
                    # Sauvegarder le maillage prédit
                    pred_vertices = pred['vertices'][i].cpu().numpy()
                    pred_faces = pred['faces'][i].cpu().numpy() if 'faces' in pred else None
                    
                    save_mesh(
                        vertices=pred_vertices,
                        faces=pred_faces,
                        filename=os.path.join(args.output_dir, f"pred_{item_ids[i]}.obj")
                    )
                    
                    # Sauvegarder le maillage de référence
                    gt_vertices = target['vertices'][i].cpu().numpy()
                    gt_faces = target['faces'][i].cpu().numpy() if 'faces' in target else None
                    
                    save_mesh(
                        vertices=gt_vertices,
                        faces=gt_faces,
                        filename=os.path.join(args.output_dir, f"gt_{item_ids[i]}.obj")
                    )
                    
                    # Visualiser la reconstruction
                    fig = visualize_reconstruction(
                        image=images[i].cpu(),
                        pred_vertices=pred_vertices,
                        gt_vertices=gt_vertices,
                        pred_faces=pred_faces,
                        gt_faces=gt_faces
                    )
                    
                    plt.savefig(os.path.join(args.output_dir, f"vis_{item_ids[i]}.png"))
                    plt.close(fig)
                    
                    visualized_samples += 1
            
            all_metrics.extend(batch_metrics)
    
    # Calculer les métriques moyennes
    mean_metrics = {}
    for metric in all_metrics[0].keys():
        if metric != 'item_id':
            values = [m[metric] for m in all_metrics]
            mean_metrics[metric] = np.mean(values)
            mean_metrics[f"{metric}_std"] = np.std(values)
    
    # Afficher les métriques moyennes
    print("\nTest Metrics:")
    for k, v in mean_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Sauvegarder les métriques
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'per_sample': all_metrics,
            'mean': mean_metrics
        }, f, indent=2)
    
    # Tracer les distributions des métriques
    metrics_to_plot = ['chamfer', 'f_score', 'normal_consistency', 'texture_similarity']
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 3*len(metrics_to_plot)))
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in all_metrics[0]:
            values = [m[metric] for m in all_metrics]
            axes[i].hist(values, bins=20)
            axes[i].axvline(mean_metrics[metric], color='r', linestyle='--')
            axes[i].set_title(f"{metric}: mean={mean_metrics[metric]:.4f}, std={mean_metrics[f'{metric}_std']:.4f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_distribution.png'))
    
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()

