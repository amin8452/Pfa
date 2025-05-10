import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import json
import trimesh
from PIL import Image
import torchvision.transforms as transforms

class GlassesDataset(Dataset):
    def __init__(self, root_dir="data/glasses", mode="train", img_size=256, augment=True):
        """
        Dataset pour les lunettes 3D
        
        Args:
            root_dir: Répertoire racine contenant les données
            mode: 'train', 'val', ou 'test'
            img_size: Taille des images d'entrée
            augment: Appliquer des augmentations de données (uniquement pour l'entraînement)
        """
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.augment = augment and mode == "train"
        
        # Lire le fichier de métadonnées
        metadata_file = os.path.join(root_dir, f"{mode}_metadata.json")
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Transformations d'image
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Charger l'image
        img_path = os.path.join(self.root_dir, item['image_path'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Charger le maillage 3D
        mesh_path = os.path.join(self.root_dir, item['mesh_path'])
        mesh = trimesh.load(mesh_path)
        
        # Normaliser les vertices
        vertices = mesh.vertices.copy()
        center = np.mean(vertices, axis=0)
        vertices = vertices - center
        scale = np.max(np.abs(vertices))
        vertices = vertices / scale
        
        # Convertir en tenseurs
        vertices_tensor = torch.FloatTensor(vertices)
        faces_tensor = torch.LongTensor(mesh.faces)
        
        # Charger la texture si disponible
        texture_tensor = None
        if 'texture_path' in item and item['texture_path']:
            texture_path = os.path.join(self.root_dir, item['texture_path'])
            if os.path.exists(texture_path):
                texture = Image.open(texture_path).convert('RGB')
                texture_transform = transforms.Compose([
                    transforms.ToTensor()
                ])
                texture_tensor = texture_transform(texture)
        
        # Créer le dictionnaire de sortie
        sample = {
            'image': image,
            'vertices': vertices_tensor,
            'faces': faces_tensor,
            'item_id': item['id']
        }
        
        if texture_tensor is not None:
            sample['texture'] = texture_tensor
            
        return sample


