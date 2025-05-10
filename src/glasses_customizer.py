import os
import sys
import argparse
import torch
import numpy as np
import trimesh
import tempfile
import json
import gradio as gr
from PIL import Image
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from material_renderer import GlassesMaterialRenderer, MaterialProperties
from virtual_tryon import VirtualTryOn

class GlassesCustomizer:
    """Tool for customizing 3D glasses models"""
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.material_renderer = GlassesMaterialRenderer()
        self.virtual_tryon = VirtualTryOn()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def load_glasses(self, glasses_path):
        """Load glasses model from file"""
        try:
            return trimesh.load(glasses_path)
        except Exception as e:
            print(f"Error loading glasses model: {e}")
            return None
    
    def customize_materials(self, mesh, material_props):
        """Apply custom materials to the glasses"""
        try:
            return self.material_renderer.apply_materials(mesh, material_props)
        except Exception as e:
            print(f"Error customizing materials: {e}")
            return mesh
    
    def change_color(self, mesh, color, part="frame"):
        """Change the color of a specific part of the glasses"""
        try:
            # Convert color to RGB if it's in hex format
            if isinstance(color, str) and color.startswith("#"):
                color = [int(color[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
            
            # Create a copy of the mesh
            mesh = mesh.copy()
            
            # Apply color to the specified part
            if part == "frame":
                # In a real implementation, you would identify which vertices belong to the frame
                # Here we'll use a simplified approach
                if hasattr(mesh.visual, "vertex_colors"):
                    # Get current vertex colors
                    vertex_colors = mesh.visual.vertex_colors.copy()
                    
                    # Identify frame vertices (simplified)
                    # In practice, you would use a more sophisticated method
                    frame_vertices = np.ones(len(mesh.vertices), dtype=bool)
                    
                    # Apply color to frame vertices
                    vertex_colors[frame_vertices, :3] = np.array(color) * 255
                    
                    # Update mesh colors
                    mesh.visual.vertex_colors = vertex_colors
                else:
                    # If no vertex colors, set material color
                    if hasattr(mesh, "visual") and hasattr(mesh.visual, "material"):
                        mesh.visual.material.baseColorFactor = color + [1.0]  # RGB + Alpha
            
            elif part == "lenses":
                # Similar approach for lenses
                if hasattr(mesh.visual, "vertex_colors"):
                    vertex_colors = mesh.visual.vertex_colors.copy()
                    
                    # Identify lens vertices (simplified)
                    lens_vertices = np.ones(len(mesh.vertices), dtype=bool)
                    
                    # Apply color to lens vertices
                    vertex_colors[lens_vertices, :3] = np.array(color) * 255
                    
                    # Update mesh colors
                    mesh.visual.vertex_colors = vertex_colors
            
            return mesh
            
        except Exception as e:
            print(f"Error changing color: {e}")
            return mesh
    
    def adjust_dimensions(self, mesh, scale_factors):
        """Adjust the dimensions of the glasses"""
        try:
            # Create a copy of the mesh
            mesh = mesh.copy()
            
            # Apply scaling
            mesh.apply_scale(scale_factors)
            
            return mesh
            
        except Exception as e:
            print(f"Error adjusting dimensions: {e}")
            return mesh
    
    def add_decoration(self, mesh, decoration_type, position):
        """Add decorative elements to the glasses"""
        try:
            # Create a copy of the mesh
            mesh = mesh.copy()
            
            # In a real implementation, you would add actual decorations
            # This is a placeholder
            print(f"Adding {decoration_type} decoration at {position}")
            
            return mesh
            
        except Exception as e:
            print(f"Error adding decoration: {e}")
            return mesh
    
    def export_customized_glasses(self, mesh, output_path=None, material_props=None):
        """Export the customized glasses to a file"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "customized_glasses.glb")
        
        try:
            if material_props:
                self.material_renderer.export_with_materials(mesh, output_path, material_props)
            else:
                mesh.export(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error exporting customized glasses: {e}")
            return None
    
    def preview_on_face(self, mesh, face_image_path, output_path=None):
        """Preview the customized glasses on a face"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "tryon_preview.jpg")
        
        try:
            # Save mesh to temporary file
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                mesh.export(tmp.name)
                mesh_path = tmp.name
            
            # Perform virtual try-on
            result = self.virtual_tryon.try_on_from_paths(face_image_path, mesh_path, output_path)
            
            # Clean up temporary file
            os.unlink(mesh_path)
            
            return result
            
        except Exception as e:
            print(f"Error previewing on face: {e}")
            return None

def create_customizer_interface(port=7861):
    """Create and launch the Gradio interface for glasses customization"""
    customizer = GlassesCustomizer()
    
    # Define Gradio interface
    with gr.Blocks(title="Glasses Customizer") as interface:
        gr.Markdown("# 3D Glasses Customizer")
        gr.Markdown("Upload a 3D glasses model and customize it")
        
        # State variables
        current_mesh = gr.State(None)
        
        with gr.Row():
            with gr.Column():
                glasses_file = gr.File(label="Upload Glasses Model (.glb, .obj)")
                
                with gr.Tab("Materials"):
                    metallic = gr.Slider(0, 1, 0.1, label="Metallic")
                    roughness = gr.Slider(0, 1, 0.5, label="Roughness")
                    transparency = gr.Slider(0, 1, 0.0, label="Transparency")
                    ior = gr.Slider(1, 2.5, 1.5, label="Index of Refraction")
                    apply_materials_btn = gr.Button("Apply Materials")
                
                with gr.Tab("Colors"):
                    frame_color = gr.ColorPicker(label="Frame Color", value="#000000")
                    lens_color = gr.ColorPicker(label="Lens Color", value="#AAAAAA")
                    apply_colors_btn = gr.Button("Apply Colors")
                
                with gr.Tab("Dimensions"):
                    width_scale = gr.Slider(0.5, 1.5, 1.0, label="Width")
                    height_scale = gr.Slider(0.5, 1.5, 1.0, label="Height")
                    depth_scale = gr.Slider(0.5, 1.5, 1.0, label="Depth")
                    apply_dimensions_btn = gr.Button("Apply Dimensions")
                
                face_image = gr.Image(type="filepath", label="Upload Face Image for Try-On")
                try_on_btn = gr.Button("Try On Glasses")
                
                export_btn = gr.Button("Export Customized Glasses")
            
            with gr.Column():
                preview = gr.Model3D(label="3D Preview")
                tryon_preview = gr.Image(label="Try-On Preview")
                download_file = gr.File(label="Download Customized Glasses")
        
        # Define functions
        def load_glasses_model(file):
            if file is None:
                return None, None
            
            try:
                mesh = customizer.load_glasses(file.name)
                
                # Export to GLB for preview
                preview_path = os.path.join(customizer.output_dir, "preview.glb")
                mesh.export(preview_path)
                
                return mesh, preview_path
            except Exception as e:
                print(f"Error loading model: {e}")
                return None, None
        
        def apply_materials_fn(mesh, metallic, roughness, transparency, ior):
            if mesh is None:
                return None
            
            try:
                material_props = MaterialProperties(
                    metallic=metallic,
                    roughness=roughness,
                    transparency=transparency,
                    ior=ior
                )
                
                mesh = customizer.customize_materials(mesh, material_props)
                
                # Export to GLB for preview
                preview_path = os.path.join(customizer.output_dir, "preview.glb")
                customizer.export_customized_glasses(mesh, preview_path, material_props)
                
                return mesh, preview_path
            except Exception as e:
                print(f"Error applying materials: {e}")
                return mesh, None
        
        def apply_colors_fn(mesh, frame_color, lens_color):
            if mesh is None:
                return None
            
            try:
                # Apply frame color
                mesh = customizer.change_color(mesh, frame_color, "frame")
                
                # Apply lens color
                mesh = customizer.change_color(mesh, lens_color, "lenses")
                
                # Export to GLB for preview
                preview_path = os.path.join(customizer.output_dir, "preview.glb")
                mesh.export(preview_path)
                
                return mesh, preview_path
            except Exception as e:
                print(f"Error applying colors: {e}")
                return mesh, None
        
        def apply_dimensions_fn(mesh, width, height, depth):
            if mesh is None:
                return None
            
            try:
                scale_factors = [width, height, depth]
                mesh = customizer.adjust_dimensions(mesh, scale_factors)
                
                # Export to GLB for preview
                preview_path = os.path.join(customizer.output_dir, "preview.glb")
                mesh.export(preview_path)
                
                return mesh, preview_path
            except Exception as e:
                print(f"Error applying dimensions: {e}")
                return mesh, None
        
        def try_on_fn(mesh, face_image):
            if mesh is None or face_image is None:
                return None
            
            try:
                result_path = customizer.preview_on_face(mesh, face_image)
                return result_path
            except Exception as e:
                print(f"Error in try-on: {e}")
                return None
        
        def export_fn(mesh):
            if mesh is None:
                return None
            
            try:
                output_path = customizer.export_customized_glasses(mesh)
                return output_path
            except Exception as e:
                print(f"Error exporting: {e}")
                return None
        
        # Connect events
        glasses_file.change(
            fn=load_glasses_model,
            inputs=[glasses_file],
            outputs=[current_mesh, preview]
        )
        
        apply_materials_btn.click(
            fn=apply_materials_fn,
            inputs=[current_mesh, metallic, roughness, transparency, ior],
            outputs=[current_mesh, preview]
        )
        
        apply_colors_btn.click(
            fn=apply_colors_fn,
            inputs=[current_mesh, frame_color, lens_color],
            outputs=[current_mesh, preview]
        )
        
        apply_dimensions_btn.click(
            fn=apply_dimensions_fn,
            inputs=[current_mesh, width_scale, height_scale, depth_scale],
            outputs=[current_mesh, preview]
        )
        
        try_on_btn.click(
            fn=try_on_fn,
            inputs=[current_mesh, face_image],
            outputs=[tryon_preview]
        )
        
        export_btn.click(
            fn=export_fn,
            inputs=[current_mesh],
            outputs=[download_file]
        )
    
    # Launch the interface
    interface.launch(server_name="0.0.0.0", server_port=port)

def parse_args():
    parser = argparse.ArgumentParser(description='Glasses customization tool')
    parser.add_argument('--port', type=int, default=7861, help='Port for the web server')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    customizer = GlassesCustomizer(output_dir=args.output_dir)
    create_customizer_interface(args.port)
