#!/usr/bin/env python
"""
Hunyuan3D-Glasses Kaggle Endpoint
---------------------------------
This script provides an endpoint for using the Hunyuan3D-Glasses pipeline in Kaggle.
It creates a simple web interface for generating 3D glasses models from 2D images.
"""

import os
import sys
import argparse
import torch
import numpy as np
import trimesh
from PIL import Image
import io
import gradio as gr
import tempfile

def parse_args():
    parser = argparse.ArgumentParser(description='Hunyuan3D-Glasses Kaggle Endpoint')
    parser.add_argument('--mode', choices=['generate', 'customize', 'tryon'], default='generate',
                        help='Mode to run')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--use_hunyuan', action='store_true', help='Use original Hunyuan3D model')
    return parser.parse_args()

def generate_glasses(image, glasses_type="eyeglasses", use_hunyuan=False, checkpoint_path=None):
    """Generate 3D glasses from an image"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save input image to temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        image_path = tmp.name
    
    try:
        if use_hunyuan:
            # Try to use Hunyuan3D model
            try:
                from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
                from hy3dgen.texgen import Hunyuan3DPaintPipeline
                
                # Load pipelines
                shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('hunyuan_model')
                texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained('hunyuan_model')
                
                # Generate 3D model
                print("Generating 3D model...")
                mesh = shape_pipeline(image=image_path)[0]
                
                # Add texture
                print("Adding texture...")
                mesh = texture_pipeline(mesh, image=image_path)
                
                # Save result
                output_path = os.path.join(output_dir, 'glasses.glb')
                mesh.export(output_path)
                
                # Create preview
                scene = trimesh.Scene(mesh)
                render = scene.save_image(resolution=[512, 512])
                preview = Image.open(io.BytesIO(render))
                preview_path = os.path.join(output_dir, 'preview.png')
                preview.save(preview_path)
                
                return output_path, preview_path
                
            except Exception as e:
                print(f"Error using Hunyuan3D: {e}")
                print("Falling back to custom model...")
                use_hunyuan = False
        
        if not use_hunyuan:
            # Use our custom model
            from src.enhanced_model_glasses import EnhancedGlassesHunyuan3DModel
            
            model = EnhancedGlassesHunyuan3DModel(checkpoint_path=checkpoint_path)
            model.to(device)
            model.eval()
            
            # Map glasses type to index
            glasses_types = ["eyeglasses", "sunglasses", "reading_glasses", "sports_glasses"]
            if glasses_type in glasses_types:
                glasses_type_idx = glasses_types.index(glasses_type)
            else:
                glasses_type_idx = 0
            
            # Create one-hot encoding
            glasses_type_tensor = torch.zeros(1, len(glasses_types))
            glasses_type_tensor[0, glasses_type_idx] = 1.0
            glasses_type_tensor = glasses_type_tensor.to(device)
            
            # Generate 3D model
            with torch.no_grad():
                result = model.generate_from_image(image_path, glasses_type_tensor)
            
            # Extract data
            vertices = result['vertices'][0].cpu().numpy()
            faces = result['faces'][0].cpu().numpy()
            texture = result['texture'][0].cpu().numpy()
            material_props = result['material_properties']
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Save texture
            texture_path = os.path.join(output_dir, 'texture.png')
            texture_img = (np.transpose(texture, (1, 2, 0)) * 255).astype(np.uint8)
            Image.fromarray(texture_img).save(texture_path)
            
            # Apply materials if available
            try:
                from src.material_renderer import GlassesMaterialRenderer, MaterialProperties
                
                renderer = GlassesMaterialRenderer()
                material_props_obj = MaterialProperties(
                    metallic=float(material_props['metallic'][0].cpu().numpy()),
                    roughness=float(material_props['roughness'][0].cpu().numpy()),
                    transparency=float(material_props['transparency'][0].cpu().numpy()),
                    ior=float(material_props['ior'][0].cpu().numpy()),
                    specular=float(material_props['specular'][0].cpu().numpy()),
                    clearcoat=float(material_props['clearcoat'][0].cpu().numpy()),
                    anisotropic=float(material_props['anisotropic'][0].cpu().numpy())
                )
                mesh = renderer.apply_materials(mesh, material_props_obj, texture_path)
            except Exception as e:
                print(f"Warning: Could not apply materials: {e}")
            
            # Save result
            output_path = os.path.join(output_dir, 'glasses.glb')
            mesh.export(output_path)
            
            # Create preview
            scene = trimesh.Scene(mesh)
            render = scene.save_image(resolution=[512, 512])
            preview = Image.open(io.BytesIO(render))
            preview_path = os.path.join(output_dir, 'preview.png')
            preview.save(preview_path)
            
            return output_path, preview_path
            
    except Exception as e:
        print(f"Error generating glasses: {e}")
        return None, None
    finally:
        # Clean up temporary file
        if os.path.exists(image_path):
            os.unlink(image_path)

def create_gradio_interface():
    """Create Gradio interface for Kaggle"""
    with gr.Blocks(title="Hunyuan3D-Glasses") as demo:
        gr.Markdown("# Reconstruction 3D de Lunettes")
        gr.Markdown("Téléchargez une image de lunettes pour générer un modèle 3D")
        
        with gr.Tab("Génération"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Image d'entrée")
                    glasses_type = gr.Dropdown(
                        choices=["eyeglasses", "sunglasses", "reading_glasses", "sports_glasses"],
                        value="eyeglasses",
                        label="Type de lunettes"
                    )
                    use_hunyuan = gr.Checkbox(label="Utiliser Hunyuan3D (si disponible)", value=True)
                    generate_btn = gr.Button("Générer le modèle 3D")
                
                with gr.Column():
                    preview_image = gr.Image(label="Aperçu")
                    model_file = gr.File(label="Télécharger le modèle 3D")
            
            generate_btn.click(
                fn=generate_glasses,
                inputs=[input_image, glasses_type, use_hunyuan],
                outputs=[model_file, preview_image]
            )
    
    return demo

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if running in Kaggle
    in_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
    if in_kaggle:
        print("Running in Kaggle environment")
    
    # Install dependencies if needed
    try:
        import gradio
    except ImportError:
        print("Installing gradio...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "gradio"])
    
    # Launch Gradio interface
    demo = create_gradio_interface()
    demo.launch(share=True)  # share=True for public link

if __name__ == "__main__":
    main()
