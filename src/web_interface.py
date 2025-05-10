import os
import sys
import argparse
import torch
import numpy as np
import trimesh
import tempfile
import base64
from PIL import Image
import io
import json
import gradio as gr

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_model_glasses import EnhancedGlassesHunyuan3DModel
from material_renderer import GlassesMaterialRenderer, MaterialProperties

# Global variables
model = None
renderer = None
output_dir = "results"

def load_model(checkpoint_path=None):
    """Load the glasses 3D reconstruction model"""
    global model
    
    if model is None:
        print("Loading model...")
        try:
            model = EnhancedGlassesHunyuan3DModel(checkpoint_path=checkpoint_path)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    
    return model

def load_renderer():
    """Load the material renderer"""
    global renderer
    
    if renderer is None:
        print("Loading renderer...")
        try:
            renderer = GlassesMaterialRenderer()
            print("Renderer loaded successfully!")
        except Exception as e:
            print(f"Error loading renderer: {e}")
            renderer = None
    
    return renderer

def process_image(image, glasses_type="eyeglasses", apply_materials=True):
    """
    Process an image and generate a 3D glasses model.
    
    Args:
        image: PIL Image or path to image
        glasses_type: Type of glasses to generate
        apply_materials: Whether to apply materials to the mesh
        
    Returns:
        Path to the generated 3D model
    """
    global model, renderer, output_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model if not loaded
    if model is None:
        model = load_model()
    
    # Load renderer if not loaded
    if renderer is None and apply_materials:
        renderer = load_renderer()
    
    # Convert glasses type to index
    glasses_types = ["eyeglasses", "sunglasses", "reading_glasses", "sports_glasses"]
    if glasses_type in glasses_types:
        glasses_type_idx = glasses_types.index(glasses_type)
    else:
        glasses_type_idx = 0  # Default to eyeglasses
    
    # Create one-hot encoding for glasses type
    glasses_type_tensor = torch.zeros(1, len(glasses_types))
    glasses_type_tensor[0, glasses_type_idx] = 1.0
    
    # Save image to temporary file if it's a PIL Image
    if isinstance(image, Image.Image):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name
    else:
        image_path = image
    
    try:
        # Generate 3D model
        with torch.no_grad():
            if torch.cuda.is_available():
                glasses_type_tensor = glasses_type_tensor.cuda()
            
            result = model.generate_from_image(image_path, glasses_type_tensor)
            
            # Extract data
            vertices = result['vertices'][0].cpu().numpy()
            faces = result['faces'][0].cpu().numpy()
            texture = result['texture'][0].cpu().numpy()
            material_props = result['material_properties']
            
            # Convert material properties to MaterialProperties object
            material_props_obj = MaterialProperties(
                metallic=float(material_props['metallic'][0].cpu().numpy()),
                roughness=float(material_props['roughness'][0].cpu().numpy()),
                transparency=float(material_props['transparency'][0].cpu().numpy()),
                ior=float(material_props['ior'][0].cpu().numpy()),
                specular=float(material_props['specular'][0].cpu().numpy()),
                clearcoat=float(material_props['clearcoat'][0].cpu().numpy()),
                anisotropic=float(material_props['anisotropic'][0].cpu().numpy())
            )
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            
            # Save texture
            texture_path = os.path.join(output_dir, "texture.png")
            texture_img = (np.transpose(texture, (1, 2, 0)) * 255).astype(np.uint8)
            Image.fromarray(texture_img).save(texture_path)
            
            # Apply materials if requested
            if apply_materials and renderer is not None:
                mesh = renderer.apply_specialized_materials(mesh, material_props_obj, glasses_type)
            
            # Export mesh
            output_path = os.path.join(output_dir, "glasses.glb")
            if renderer is not None:
                renderer.export_with_materials(mesh, output_path, material_props_obj, texture_path)
            else:
                mesh.export(output_path)
            
            return output_path
            
    except Exception as e:
        print(f"Error generating 3D model: {e}")
        return None
    finally:
        # Clean up temporary file if created
        if isinstance(image, Image.Image) and os.path.exists(image_path):
            os.unlink(image_path)

def generate_3d_glasses(image, glasses_type, apply_materials):
    """
    Gradio interface function to generate 3D glasses.
    
    Args:
        image: Input image
        glasses_type: Type of glasses
        apply_materials: Whether to apply materials
        
    Returns:
        Tuple of (model_path, preview_image)
    """
    try:
        # Process the image
        model_path = process_image(image, glasses_type, apply_materials)
        
        if model_path is None:
            return None, "Failed to generate 3D model"
        
        # Create a preview image
        mesh = trimesh.load(model_path)
        scene = trimesh.Scene(mesh)
        
        # Render from multiple angles
        angles = [0, 45, 90, 180]
        previews = []
        
        for angle in angles:
            # Rotate the camera
            camera_transform = trimesh.transformations.rotation_matrix(
                angle * np.pi / 180, [0, 1, 0])
            
            # Render
            preview = scene.save_image(resolution=[512, 512], transform=camera_transform)
            previews.append(Image.open(io.BytesIO(preview)))
        
        # Combine previews into a grid
        width, height = previews[0].size
        grid = Image.new('RGB', (width * 2, height * 2))
        
        for i, preview in enumerate(previews):
            x = (i % 2) * width
            y = (i // 2) * height
            grid.paste(preview, (x, y))
        
        # Save the grid
        preview_path = os.path.join(output_dir, "preview.png")
        grid.save(preview_path)
        
        return model_path, preview_path
        
    except Exception as e:
        print(f"Error in generate_3d_glasses: {e}")
        return None, f"Error: {str(e)}"

def create_web_interface(checkpoint_path=None, port=7860):
    """Create and launch the Gradio web interface"""
    global output_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    load_model(checkpoint_path)
    
    # Load renderer
    load_renderer()
    
    # Define Gradio interface
    with gr.Blocks(title="3D Glasses Generator") as interface:
        gr.Markdown("# 3D Glasses Generator")
        gr.Markdown("Upload an image of glasses to generate a 3D model")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                glasses_type = gr.Dropdown(
                    choices=["eyeglasses", "sunglasses", "reading_glasses", "sports_glasses"],
                    value="eyeglasses",
                    label="Glasses Type"
                )
                apply_materials = gr.Checkbox(value=True, label="Apply Realistic Materials")
                generate_button = gr.Button("Generate 3D Model")
            
            with gr.Column():
                preview_image = gr.Image(label="Preview")
                model_file = gr.File(label="Download 3D Model")
        
        generate_button.click(
            fn=generate_3d_glasses,
            inputs=[input_image, glasses_type, apply_materials],
            outputs=[model_file, preview_image]
        )
    
    # Launch the interface
    interface.launch(server_name="0.0.0.0", server_port=port)

def parse_args():
    parser = argparse.ArgumentParser(description='Web interface for 3D glasses generation')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--port', type=int, default=7860, help='Port for the web server')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir
    create_web_interface(args.checkpoint, args.port)
