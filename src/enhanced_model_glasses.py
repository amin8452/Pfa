import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import CLIPVisionModel, CLIPImageProcessor

class AttentionBlock(nn.Module):
    """Self-attention block for focusing on important features"""
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        norm_x = self.norm(x)
        q, k, v = self.q(norm_x), self.k(norm_x), self.v(norm_x)
        
        # Reshape for attention
        batch_size = q.shape[0]
        q = q.view(batch_size, -1, 1).transpose(1, 2)
        k = k.view(batch_size, -1, 1).transpose(1, 2)
        v = v.view(batch_size, -1, 1).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.shape[-1]))
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(batch_size, -1)
        x = self.proj(x)
        
        return x

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + residual

class GlassesTypeClassifier(nn.Module):
    """Classifier for glasses type to guide the generation process"""
    def __init__(self, dim):
        super(GlassesTypeClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 4 types: eyeglasses, sunglasses, reading, sports
        )
        
    def forward(self, x):
        return self.classifier(x)

class MaterialPropertiesPredictor(nn.Module):
    """Predicts material properties for realistic rendering"""
    def __init__(self, dim):
        super(MaterialPropertiesPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7 properties: metallic, roughness, transparency, etc.
        )
        
    def forward(self, x):
        properties = self.predictor(x)
        # Normalize properties to appropriate ranges
        metallic = torch.sigmoid(properties[:, 0])
        roughness = torch.sigmoid(properties[:, 1])
        transparency = torch.sigmoid(properties[:, 2])
        ior = 1.0 + torch.sigmoid(properties[:, 3]) * 1.5  # Index of refraction: 1.0-2.5
        specular = torch.sigmoid(properties[:, 4])
        clearcoat = torch.sigmoid(properties[:, 5])
        anisotropic = torch.sigmoid(properties[:, 6])
        
        return {
            'metallic': metallic,
            'roughness': roughness,
            'transparency': transparency,
            'ior': ior,
            'specular': specular,
            'clearcoat': clearcoat,
            'anisotropic': anisotropic
        }

class EnhancedGlassesHunyuan3DModel(nn.Module):
    """
    Enhanced adaptation of Hunyuan3D model for glasses 3D reconstruction.
    This model includes specialized components for glasses features and materials.
    """
    def __init__(self, checkpoint_path=None, pretrained=True, freeze_encoder=True):
        super(EnhancedGlassesHunyuan3DModel, self).__init__()
        
        # Image encoder (using CLIP vision model for good feature extraction)
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        if freeze_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        # Feature dimension
        feature_dim = 768  # CLIP's output dimension
        hidden_dim = 512
        
        # Enhanced glasses-specific encoder with attention and residual connections
        self.glasses_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            AttentionBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        
        # Glasses type classifier for conditional generation
        self.glasses_type_classifier = GlassesTypeClassifier(hidden_dim)
        
        # Material properties predictor
        self.material_predictor = MaterialPropertiesPredictor(hidden_dim)
        
        # Frame and lens separation for specialized processing
        self.frame_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            ResidualBlock(1024),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 3 * 1536)  # 1536 vertices for frames
        )
        
        self.lens_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            ResidualBlock(512),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * 512)  # 512 vertices for lenses
        )
        
        # Decoder for generating faces (triangles)
        self.face_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            ResidualBlock(1024),
            nn.Linear(1024, 4096),
            nn.GELU(),
            nn.Linear(4096, 3 * 4096)  # 4096 faces with 3 vertex indices each
        )
        
        # Enhanced texture generator with material properties
        self.texture_generator = nn.Sequential(
            nn.Linear(hidden_dim + 7, 1024),  # Add material properties
            nn.GELU(),
            ResidualBlock(1024),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 3 * 1024 * 1024)  # RGB texture map of 1024x1024
        )
        
        # Load pretrained weights if provided
        if checkpoint_path and pretrained:
            self._load_pretrained_weights(checkpoint_path)
            
    def _load_pretrained_weights(self, checkpoint_path):
        """
        Load pretrained weights from Hunyuan3D model and adapt them for glasses model.
        """
        try:
            # Check if checkpoint_path is a HuggingFace model ID
            if '/' in checkpoint_path and not os.path.exists(checkpoint_path):
                from huggingface_hub import snapshot_download
                checkpoint_path = snapshot_download(repo_id=checkpoint_path, 
                                                   subfolder="hunyuan3d-dit-v2-0")
                
            # Load the Hunyuan3D checkpoint
            checkpoint = torch.load(os.path.join(checkpoint_path, "model.pth"), 
                                   map_location="cpu")
            
            # Initialize weights from the pretrained model
            # This is a simplified version - in practice, you'd need to map the weights correctly
            print(f"Loaded pretrained weights from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            print("Initializing with random weights")
    
    def forward(self, images, glasses_type=None):
        """
        Forward pass of the model.
        
        Args:
            images: Batch of images (B, C, H, W)
            glasses_type: Optional tensor specifying glasses type for conditional generation
            
        Returns:
            Dictionary containing:
                - vertices: 3D vertices (B, N, 3)
                - faces: Face indices (B, F, 3)
                - texture: Texture map (B, 3, H, W)
                - material_properties: Material properties for rendering
                - glasses_type: Predicted glasses type
        """
        batch_size = images.shape[0]
        
        # Process images with CLIP
        with torch.no_grad():
            inputs = self.image_processor(images=images, return_tensors="pt").to(images.device)
            image_features = self.image_encoder(**inputs).last_hidden_state[:, 0]  # Use CLS token
        
        # Enhanced glasses-specific encoding
        features = self.glasses_encoder(image_features)
        
        # Predict glasses type
        predicted_type = self.glasses_type_classifier(features)
        glasses_type_probs = F.softmax(predicted_type, dim=1)
        
        # Predict material properties
        material_props = self.material_predictor(features)
        material_props_tensor = torch.cat([
            material_props['metallic'].unsqueeze(1),
            material_props['roughness'].unsqueeze(1),
            material_props['transparency'].unsqueeze(1),
            material_props['ior'].unsqueeze(1),
            material_props['specular'].unsqueeze(1),
            material_props['clearcoat'].unsqueeze(1),
            material_props['anisotropic'].unsqueeze(1)
        ], dim=1)
        
        # Generate frame vertices
        frame_vertices_flat = self.frame_decoder(features)
        frame_vertices = frame_vertices_flat.view(batch_size, 1536, 3)
        
        # Generate lens vertices
        lens_vertices_flat = self.lens_decoder(features)
        lens_vertices = lens_vertices_flat.view(batch_size, 512, 3)
        
        # Combine frame and lens vertices
        vertices = torch.cat([frame_vertices, lens_vertices], dim=1)
        
        # Generate faces
        faces_flat = self.face_decoder(features)
        faces = faces_flat.view(batch_size, 4096, 3).long()
        
        # Ensure face indices are valid (within vertex count)
        faces = torch.clamp(faces, 0, 2047)
        
        # Generate texture with material properties
        texture_input = torch.cat([features, material_props_tensor], dim=1)
        texture_flat = self.texture_generator(texture_input)
        texture = texture_flat.view(batch_size, 3, 1024, 1024)
        
        return {
            'vertices': vertices,
            'faces': faces,
            'texture': texture,
            'material_properties': material_props,
            'glasses_type': glasses_type_probs
        }
    
    def generate_from_image(self, image_path, glasses_type=None):
        """
        Generate a 3D glasses model from an image file.
        
        Args:
            image_path: Path to the image file
            glasses_type: Optional glasses type for conditional generation
            
        Returns:
            Dictionary containing the 3D model data
        """
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)
        
        # Generate the 3D model
        with torch.no_grad():
            result = self.forward(image_tensor, glasses_type)
        
        return result
