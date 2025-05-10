import torch
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import torch.nn.functional as F

def chamfer_distance(pred_points, gt_points, bidirectional=True):
    """
    Calcule la distance de Chamfer entre deux nuages de points
    
    Args:
        pred_points: Points prédits (B, N, 3)
        gt_points: Points de référence (B, M, 3)
        bidirectional: Si True, calcule la distance dans les deux sens
        
    Returns:
        Distance de Chamfer moyenne
    """
    batch_size = pred_points.shape[0]
    
    # Pour chaque point dans pred_points, trouver le point le plus proche dans gt_points
    dist1 = torch.cdist(pred_points, gt_points)
    min_dist1, _ = torch.min(dist1, dim=2)  # (B, N)
    
    if bidirectional:
        # Pour chaque point dans gt_points, trouver le point le plus proche dans pred_points
        min_dist2, _ = torch.min(dist1, dim=1)  # (B, M)
        chamfer = torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1)  # (B,)
    else:
        chamfer = torch.mean(min_dist1, dim=1)  # (B,)
    
    return torch.mean(chamfer)

def f_score(pred_points, gt_points, threshold=0.01):
    """
    Calcule le F-score entre deux nuages de points
    
    Args:
        pred_points: Points prédits (B, N, 3)
        gt_points: Points de référence (B, M, 3)
        threshold: Seuil de distance pour considérer un point comme correctement prédit
        
    Returns:
        F-score moyen
    """
    batch_size = pred_points.shape[0]
    f_scores = []
    
    for i in range(batch_size):
        # Calculer les distances
        dist1 = torch.cdist(pred_points[i:i+1], gt_points[i:i+1])[0]
        dist2 = torch.cdist(gt_points[i:i+1], pred_points[i:i+1])[0]
        
        # Précision: proportion de points prédits proches d'un point de référence
        precision = torch.mean((torch.min(dist1, dim=1)[0] < threshold).float())
        
        # Rappel: proportion de points de référence proches d'un point prédit
        recall = torch.mean((torch.min(dist2, dim=1)[0] < threshold).float())
        
        # F-score
        if precision + recall > 0:
            f_score = 2 * precision * recall / (precision + recall)
        else:
            f_score = torch.tensor(0.0, device=pred_points.device)
        
        f_scores.append(f_score)
    
    return torch.mean(torch.stack(f_scores))

def normal_consistency(pred_vertices, pred_faces, gt_vertices, gt_faces):
    """
    Calcule la cohérence des normales entre deux maillages
    
    Args:
        pred_vertices: Vertices prédits (B, N, 3)
        pred_faces: Faces prédites (B, F, 3)
        gt_vertices: Vertices de référence (B, M, 3)
        gt_faces: Faces de référence (B, G, 3)
        
    Returns:
        Score de cohérence des normales moyen
    """
    batch_size = pred_vertices.shape[0]
    consistency_scores = []
    
    for i in range(batch_size):
        # Calculer les normales des faces prédites
        v1 = pred_vertices[i, pred_faces[i, :, 0]]
        v2 = pred_vertices[i, pred_faces[i, :, 1]]
        v3 = pred_vertices[i, pred_faces[i, :, 2]]
        pred_normals = torch.cross(v2 - v1, v3 - v1)
        pred_normals = F.normalize(pred_normals, dim=1)
        
        # Calculer les normales des faces de référence
        v1 = gt_vertices[i, gt_faces[i, :, 0]]
        v2 = gt_vertices[i, gt_faces[i, :, 1]]
        v3 = gt_vertices[i, gt_faces[i, :, 2]]
        gt_normals = torch.cross(v2 - v1, v3 - v1)
        gt_normals = F.normalize(gt_normals, dim=1)
        
        # Trouver les correspondances entre les faces
        pred_centers = (v1 + v2 + v3) / 3
        gt_centers = (v1 + v2 + v3) / 3
        
        # Pour chaque face prédite, trouver la face de référence la plus proche
        dist = torch.cdist(pred_centers, gt_centers)
        min_idx = torch.argmin(dist, dim=1)
        
        # Calculer la cohérence des normales (produit scalaire des normales)
        corresponding_gt_normals = gt_normals[min_idx]
        consistency = torch.abs(torch.sum(pred_normals * corresponding_gt_normals, dim=1))
        
        consistency_scores.append(torch.mean(consistency))
    
    return torch.mean(torch.stack(consistency_scores))

def texture_similarity(pred_texture, gt_texture):
    """
    Calcule la similarité entre les textures prédites et de référence
    
    Args:
        pred_texture: Texture prédite (B, C, H, W)
        gt_texture: Texture de référence (B, C, H, W)
        
    Returns:
        Score de similarité moyen (SSIM ou autre métrique)
    """
    if pred_texture is None or gt_texture is None:
        return torch.tensor(0.0)
    
    # Redimensionner si nécessaire
    if pred_texture.shape != gt_texture.shape:
        pred_texture = F.interpolate(pred_texture, size=gt_texture.shape[2:], mode='bilinear')
    
    # Calculer l'erreur quadratique moyenne
    mse = torch.mean((pred_texture - gt_texture) ** 2, dim=[1, 2, 3])
    
    # Convertir en PSNR (Peak Signal-to-Noise Ratio)
    psnr = 10 * torch.log10(1.0 / mse)
    
    return torch.mean(psnr)

def evaluate_glasses_reconstruction(pred_data, gt_data):
    """
    Évalue la qualité de la reconstruction 3D des lunettes
    
    Args:
        pred_data: Dictionnaire contenant les données prédites
        gt_data: Dictionnaire contenant les données de référence
        
    Returns:
        Dictionnaire des métriques
    """
    metrics = {}
    
    # Chamfer Distance
    metrics['chamfer'] = chamfer_distance(pred_data['vertices'], gt_data['vertices']).item()
    
    # F-score
    metrics['f_score'] = f_score(pred_data['vertices'], gt_data['vertices']).item()
    
    # Normal consistency si les faces sont disponibles
    if 'faces' in pred_data and 'faces' in gt_data:
        metrics['normal_consistency'] = normal_consistency(
            pred_data['vertices'], pred_data['faces'],
            gt_data['vertices'], gt_data['faces']
        ).item()
    
    # Texture similarity si les textures sont disponibles
    if 'texture' in pred_data and 'texture' in gt_data:
        metrics['texture_similarity'] = texture_similarity(
            pred_data['texture'], gt_data['texture']
        ).item()
    
    # Métrique spécifique aux lunettes: précision des branches
    # Cette métrique pourrait être personnalisée en fonction de la géométrie des lunettes
    
    return metrics


