import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import trimesh

from model_glasses import GlassesHunyuan3DModel
from utils import save_mesh, visualize_reconstruction

def parse_args():
    parser = argparse.ArgumentParser(description='Demo for 3D glasses reconstruction')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to model checkpoint or HuggingFace model ID')
    parser.add_argument('--use_hunyuan', action='store_true', 
                        help='Use original Hunyuan3D model as base')
    parser.add_argument('--visualize', action='store_true', help='Visualize the result')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    if args.use_hunyuan:
        print("Using Hunyuan3D model as base...")
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            
            # Load Hunyuan3D pipeline
            shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
            texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
            
            # Generate 3D model
            print("Generating 3D model from image...")
            mesh = shape_pipeline(image=args.image)[0]
            
            # Add texture
            print("Adding texture...")
            mesh = texture_pipeline(mesh, image=args.image)
            
            # Save the result
            output_path = os.path.join(args.output_dir, 'glasses_hunyuan.glb')
            mesh.export(output_path)
            print(f"Saved 3D model to {output_path}")
            
            # Visualize if requested
            if args.visualize:
                # Convert to numpy arrays for visualization
                vertices = np.array(mesh.vertices)
                faces = np.array(mesh.faces)
                
                # Load the input image
                image = Image.open(args.image).convert('RGB')
                
                # Visualize
                fig = visualize_reconstruction(
                    image=np.array(image) / 255.0,
                    pred_vertices=vertices,
                    pred_faces=faces,
                    gt_vertices=None,
                    gt_faces=None
                )
                
                plt.savefig(os.path.join(args.output_dir, 'visualization_hunyuan.png'))
                plt.close(fig)
                print(f"Saved visualization to {os.path.join(args.output_dir, 'visualization_hunyuan.png')}")
            
            return
        except ImportError:
            print("Failed to import Hunyuan3D modules. Falling back to our custom model.")
    
    # Initialize our custom model
    model = GlassesHunyuan3DModel(checkpoint_path=args.checkpoint)
    model.to(device)
    model.eval()
    
    # Load and preprocess the image
    image = Image.open(args.image).convert('RGB')
    
    # Generate 3D model
    print("Generating 3D model from image...")
    result = model.generate_from_image(args.image)
    
    # Convert tensors to numpy arrays
    vertices = result['vertices'][0].cpu().numpy()
    faces = result['faces'][0].cpu().numpy()
    texture = result['texture'][0].cpu().numpy()
    
    # Save the mesh
    output_path = os.path.join(args.output_dir, 'glasses.obj')
    texture_path = os.path.join(args.output_dir, 'glasses_texture.png')
    
    # Save texture as image
    texture_img = (np.transpose(texture, (1, 2, 0)) * 255).astype(np.uint8)
    Image.fromarray(texture_img).save(texture_path)
    
    # Save mesh
    save_mesh(
        vertices=vertices,
        faces=faces,
        filename=output_path,
        texture=texture_img,
        texture_filename=texture_path
    )
    
    print(f"Saved 3D model to {output_path}")
    print(f"Saved texture to {texture_path}")
    
    # Visualize if requested
    if args.visualize:
        fig = visualize_reconstruction(
            image=np.array(image) / 255.0,
            pred_vertices=vertices,
            pred_faces=faces,
            gt_vertices=None,
            gt_faces=None
        )
        
        plt.savefig(os.path.join(args.output_dir, 'visualization.png'))
        plt.close(fig)
        print(f"Saved visualization to {os.path.join(args.output_dir, 'visualization.png')}")

if __name__ == '__main__':
    main()
