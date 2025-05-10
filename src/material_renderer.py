import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
import os
from PIL import Image
import json

class MaterialProperties:
    """Class to store and manipulate material properties for glasses"""
    def __init__(self, 
                 metallic=0.0, 
                 roughness=0.5, 
                 transparency=0.0, 
                 ior=1.5, 
                 specular=0.5, 
                 clearcoat=0.0, 
                 anisotropic=0.0):
        self.metallic = metallic
        self.roughness = roughness
        self.transparency = transparency
        self.ior = ior
        self.specular = specular
        self.clearcoat = clearcoat
        self.anisotropic = anisotropic
    
    @classmethod
    def from_dict(cls, props_dict):
        """Create MaterialProperties from a dictionary"""
        return cls(
            metallic=props_dict.get('metallic', 0.0),
            roughness=props_dict.get('roughness', 0.5),
            transparency=props_dict.get('transparency', 0.0),
            ior=props_dict.get('ior', 1.5),
            specular=props_dict.get('specular', 0.5),
            clearcoat=props_dict.get('clearcoat', 0.0),
            anisotropic=props_dict.get('anisotropic', 0.0)
        )
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'metallic': float(self.metallic),
            'roughness': float(self.roughness),
            'transparency': float(self.transparency),
            'ior': float(self.ior),
            'specular': float(self.specular),
            'clearcoat': float(self.clearcoat),
            'anisotropic': float(self.anisotropic)
        }
    
    def to_tensor(self):
        """Convert to tensor"""
        return torch.tensor([
            self.metallic,
            self.roughness,
            self.transparency,
            self.ior,
            self.specular,
            self.clearcoat,
            self.anisotropic
        ])
    
    @classmethod
    def for_glasses_type(cls, glasses_type):
        """Create default material properties based on glasses type"""
        if glasses_type == 'eyeglasses':
            return cls(
                metallic=0.1,
                roughness=0.2,
                transparency=0.9,
                ior=1.5,
                specular=0.8,
                clearcoat=0.5,
                anisotropic=0.0
            )
        elif glasses_type == 'sunglasses':
            return cls(
                metallic=0.3,
                roughness=0.3,
                transparency=0.4,
                ior=1.5,
                specular=0.7,
                clearcoat=0.7,
                anisotropic=0.1
            )
        elif glasses_type == 'reading_glasses':
            return cls(
                metallic=0.05,
                roughness=0.2,
                transparency=0.95,
                ior=1.52,
                specular=0.6,
                clearcoat=0.3,
                anisotropic=0.0
            )
        elif glasses_type == 'sports_glasses':
            return cls(
                metallic=0.2,
                roughness=0.4,
                transparency=0.7,
                ior=1.48,
                specular=0.5,
                clearcoat=0.2,
                anisotropic=0.0
            )
        else:
            return cls()  # Default properties

class GlassesMaterialRenderer:
    """
    Renderer for glasses with realistic materials.
    This class applies material properties to the glasses mesh.
    """
    def __init__(self):
        pass
    
    def apply_materials(self, mesh, material_props, texture_path=None):
        """
        Apply material properties to the mesh.
        
        Args:
            mesh: A trimesh object
            material_props: MaterialProperties object or dictionary
            texture_path: Optional path to texture image
            
        Returns:
            Updated mesh with materials applied
        """
        if isinstance(material_props, dict):
            material_props = MaterialProperties.from_dict(material_props)
        
        # Create a copy of the mesh to avoid modifying the original
        mesh = mesh.copy()
        
        # Apply material properties to the mesh
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
            # Apply texture if provided
            if texture_path and os.path.exists(texture_path):
                mesh.visual.material.image = texture_path
            
            # Apply material properties
            mesh.visual.material.metallicFactor = material_props.metallic
            mesh.visual.material.roughnessFactor = material_props.roughness
            
            # Handle transparency
            if material_props.transparency > 0.01:
                mesh.visual.material.alphaMode = 'BLEND'
                mesh.visual.material.baseColorFactor = [1.0, 1.0, 1.0, 1.0 - material_props.transparency]
            
            # Apply other properties
            mesh.visual.material.specularFactor = material_props.specular
            
            # Some properties might not be directly supported by trimesh
            # We'll store them as custom properties
            custom_props = {
                'ior': material_props.ior,
                'clearcoat': material_props.clearcoat,
                'anisotropic': material_props.anisotropic
            }
            
            # Store custom properties in the mesh metadata
            if not hasattr(mesh, 'metadata'):
                mesh.metadata = {}
            mesh.metadata['material_properties'] = custom_props
        
        return mesh
    
    def separate_frame_and_lenses(self, mesh, vertices, faces):
        """
        Separate the mesh into frame and lenses based on material properties.
        
        Args:
            mesh: Original mesh
            vertices: Vertices tensor
            faces: Faces tensor
            
        Returns:
            Tuple of (frame_mesh, lenses_mesh)
        """
        # This is a simplified implementation
        # In practice, you would use more sophisticated methods to separate parts
        
        # Convert tensors to numpy arrays if needed
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()
        
        # Assume the first 1536 vertices are for the frame and the rest are for lenses
        # This should match the model's output structure
        frame_vertices = vertices[:1536]
        lens_vertices = vertices[1536:]
        
        # Create separate meshes
        # This is simplified - in practice you'd need to handle faces correctly
        frame_mesh = trimesh.Trimesh(vertices=frame_vertices, process=False)
        lenses_mesh = trimesh.Trimesh(vertices=lens_vertices, process=False)
        
        return frame_mesh, lenses_mesh
    
    def apply_specialized_materials(self, mesh, material_props, glasses_type):
        """
        Apply specialized materials based on glasses type.
        
        Args:
            mesh: Original mesh
            material_props: MaterialProperties object
            glasses_type: Type of glasses (eyeglasses, sunglasses, etc.)
            
        Returns:
            Mesh with specialized materials
        """
        # Separate frame and lenses
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        # In a real implementation, you would use a more sophisticated method
        # to identify which vertices belong to the frame vs. lenses
        # Here we'll use a simplified approach based on the y-coordinate
        
        # Find the center of the mesh
        center = np.mean(vertices, axis=0)
        
        # Identify lens vertices (typically in the center of the glasses)
        lens_mask = np.linalg.norm(vertices[:, :2] - center[:2], axis=1) < np.percentile(
            np.linalg.norm(vertices[:, :2] - center[:2], axis=1), 30)
        
        # Create submeshes
        lens_vertices = vertices[lens_mask]
        frame_vertices = vertices[~lens_mask]
        
        # Create lens material
        lens_material = MaterialProperties(
            metallic=0.0,
            roughness=0.1,
            transparency=0.9 if glasses_type != 'sunglasses' else 0.3,
            ior=1.5,
            specular=0.9,
            clearcoat=0.8,
            anisotropic=0.0
        )
        
        # Create frame material
        frame_material = MaterialProperties(
            metallic=material_props.metallic,
            roughness=material_props.roughness,
            transparency=0.0,
            ior=1.0,
            specular=material_props.specular,
            clearcoat=material_props.clearcoat,
            anisotropic=material_props.anisotropic
        )
        
        # In a real implementation, you would create separate meshes
        # and apply different materials to each
        # For this example, we'll just return the original mesh with the frame material
        return self.apply_materials(mesh, frame_material)
    
    def export_with_materials(self, mesh, output_path, material_props=None, texture_path=None):
        """
        Export the mesh with materials to a file.
        
        Args:
            mesh: Mesh to export
            output_path: Path to save the mesh
            material_props: Optional MaterialProperties
            texture_path: Optional path to texture
            
        Returns:
            Path to the exported file
        """
        if material_props:
            mesh = self.apply_materials(mesh, material_props, texture_path)
        
        # Export the mesh
        mesh.export(output_path)
        
        # If it's a glTF/GLB format, also export material properties as JSON
        if output_path.endswith('.gltf') or output_path.endswith('.glb'):
            props_path = os.path.splitext(output_path)[0] + '_materials.json'
            if material_props:
                with open(props_path, 'w') as f:
                    if isinstance(material_props, MaterialProperties):
                        json.dump(material_props.to_dict(), f, indent=2)
                    else:
                        json.dump(material_props, f, indent=2)
        
        return output_path

# Example usage:
# renderer = GlassesMaterialRenderer()
# material_props = MaterialProperties.for_glasses_type('eyeglasses')
# mesh = renderer.apply_materials(mesh, material_props, 'texture.png')
# renderer.export_with_materials(mesh, 'glasses.glb', material_props)
