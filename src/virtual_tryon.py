import os
import sys
import argparse
import torch
import numpy as np
import trimesh
import cv2
import mediapipe as mp
from PIL import Image
import tempfile
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from material_renderer import GlassesMaterialRenderer, MaterialProperties

class FaceLandmarkDetector:
    """Detect face landmarks for glasses placement"""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def detect_landmarks(self, image):
        """
        Detect face landmarks in an image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with face landmarks and measurements
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Convert to RGB if needed
        if image_np.shape[-1] == 4:  # RGBA
            image_np = image_np[..., :3]
        
        # Process the image
        results = self.face_mesh.process(image_np)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Extract key points for glasses placement
        # These indices are based on MediaPipe Face Mesh topology
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        
        # Key points for glasses placement
        left_eye = np.mean([[landmarks[p].x, landmarks[p].y, landmarks[p].z] 
                           for p in [33, 133, 173, 157, 158, 159, 160, 161, 246]], axis=0)
        
        right_eye = np.mean([[landmarks[p].x, landmarks[p].y, landmarks[p].z] 
                            for p in [362, 263, 466, 388, 387, 386, 385, 384, 398]], axis=0)
        
        nose_bridge = np.mean([[landmarks[p].x, landmarks[p].y, landmarks[p].z] 
                              for p in [168, 6, 197, 195, 5]], axis=0)
        
        left_temple = np.mean([[landmarks[p].x, landmarks[p].y, landmarks[p].z] 
                              for p in [234, 93, 132, 58, 172]], axis=0)
        
        right_temple = np.mean([[landmarks[p].x, landmarks[p].y, landmarks[p].z] 
                               for p in [454, 323, 361, 288, 397]], axis=0)
        
        # Calculate face measurements
        eye_distance = np.linalg.norm(right_eye[:2] - left_eye[:2])
        face_width = np.linalg.norm(right_temple[:2] - left_temple[:2])
        
        # Image dimensions for normalization
        height, width = image_np.shape[:2]
        
        # Normalize coordinates to image dimensions
        left_eye[0] *= width
        left_eye[1] *= height
        right_eye[0] *= width
        right_eye[1] *= height
        nose_bridge[0] *= width
        nose_bridge[1] *= height
        left_temple[0] *= width
        left_temple[1] *= height
        right_temple[0] *= width
        right_temple[1] *= height
        
        # Calculate face angle (rotation around y-axis)
        # This is a simplified calculation
        dx = right_eye[0] - left_eye[0]
        dz = right_eye[2] - left_eye[2]
        face_angle = np.arctan2(dz, dx) * 180 / np.pi
        
        return {
            'left_eye': left_eye.tolist(),
            'right_eye': right_eye.tolist(),
            'nose_bridge': nose_bridge.tolist(),
            'left_temple': left_temple.tolist(),
            'right_temple': right_temple.tolist(),
            'eye_distance': float(eye_distance * width),
            'face_width': float(face_width * width),
            'face_angle': float(face_angle),
            'image_width': width,
            'image_height': height
        }

class GlassesAdjuster:
    """Adjust 3D glasses to fit a face"""
    def __init__(self):
        pass
    
    def adjust_glasses(self, glasses_mesh, face_landmarks):
        """
        Adjust glasses mesh to fit the detected face.
        
        Args:
            glasses_mesh: Trimesh object of glasses
            face_landmarks: Dictionary with face landmarks
            
        Returns:
            Adjusted glasses mesh
        """
        if face_landmarks is None:
            return glasses_mesh
        
        # Create a copy of the mesh to avoid modifying the original
        adjusted_mesh = glasses_mesh.copy()
        
        # Extract face measurements
        eye_distance = face_landmarks['eye_distance']
        face_width = face_landmarks['face_width']
        face_angle = face_landmarks['face_angle']
        
        # Calculate scaling factor based on eye distance
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        current_width = adjusted_mesh.bounds[1, 0] - adjusted_mesh.bounds[0, 0]
        scale_factor = face_width / current_width * 0.9  # Slightly smaller than face width
        
        # Scale the mesh
        adjusted_mesh.apply_scale(scale_factor)
        
        # Calculate translation to position glasses on the face
        nose_bridge = np.array(face_landmarks['nose_bridge'])
        
        # Center the glasses on the nose bridge
        center = adjusted_mesh.centroid
        translation = nose_bridge - center
        
        # Adjust depth (z-coordinate)
        # In a real implementation, you'd use the face geometry to determine proper depth
        translation[2] = 0  # Simplified
        
        # Apply translation
        adjusted_mesh.apply_translation(translation)
        
        # Apply rotation based on face angle
        rotation = trimesh.transformations.rotation_matrix(
            np.radians(face_angle), [0, 1, 0], adjusted_mesh.centroid)
        adjusted_mesh.apply_transform(rotation)
        
        return adjusted_mesh

class VirtualTryOn:
    """Virtual try-on system for glasses"""
    def __init__(self):
        self.landmark_detector = FaceLandmarkDetector()
        self.glasses_adjuster = GlassesAdjuster()
        self.material_renderer = GlassesMaterialRenderer()
    
    def try_on_glasses(self, face_image, glasses_mesh, output_path=None):
        """
        Perform virtual try-on of glasses on a face image.
        
        Args:
            face_image: PIL Image or path to face image
            glasses_mesh: Trimesh object or path to glasses mesh
            output_path: Path to save the result
            
        Returns:
            PIL Image with glasses overlaid on the face
        """
        # Load face image if path is provided
        if isinstance(face_image, str):
            face_image = Image.open(face_image).convert('RGB')
        
        # Load glasses mesh if path is provided
        if isinstance(glasses_mesh, str):
            glasses_mesh = trimesh.load(glasses_mesh)
        
        # Detect face landmarks
        landmarks = self.landmark_detector.detect_landmarks(face_image)
        
        if landmarks is None:
            print("No face detected in the image")
            return face_image
        
        # Adjust glasses to fit the face
        adjusted_glasses = self.glasses_adjuster.adjust_glasses(glasses_mesh, landmarks)
        
        # Create a scene with the adjusted glasses
        scene = trimesh.Scene(adjusted_glasses)
        
        # Set up camera parameters based on face orientation
        # This is a simplified approach
        camera_transform = np.eye(4)
        camera_fov = 60.0
        
        # Render the glasses
        rendered_glasses = scene.save_image(
            resolution=[face_image.width, face_image.height],
            transform=camera_transform,
            fov=camera_fov
        )
        
        # Convert rendered image to PIL
        glasses_image = Image.open(io.BytesIO(rendered_glasses)).convert('RGBA')
        
        # Overlay glasses on face image
        result_image = Image.new('RGBA', face_image.size)
        result_image.paste(face_image.convert('RGBA'), (0, 0))
        result_image.alpha_composite(glasses_image)
        
        # Save result if output path is provided
        if output_path:
            result_image.convert('RGB').save(output_path)
        
        return result_image
    
    def try_on_from_paths(self, face_image_path, glasses_model_path, output_path):
        """
        Perform virtual try-on using file paths.
        
        Args:
            face_image_path: Path to face image
            glasses_model_path: Path to glasses 3D model
            output_path: Path to save the result
            
        Returns:
            Path to the result image
        """
        try:
            # Load face image
            face_image = Image.open(face_image_path).convert('RGB')
            
            # Load glasses model
            glasses_mesh = trimesh.load(glasses_model_path)
            
            # Perform try-on
            result = self.try_on_glasses(face_image, glasses_mesh, output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error in virtual try-on: {e}")
            return None

def parse_args():
    parser = argparse.ArgumentParser(description='Virtual try-on for 3D glasses')
    parser.add_argument('--face', type=str, required=True, help='Path to face image')
    parser.add_argument('--glasses', type=str, required=True, help='Path to glasses 3D model')
    parser.add_argument('--output', type=str, default='tryon_result.jpg', help='Path to save the result')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    tryon = VirtualTryOn()
    result_path = tryon.try_on_from_paths(args.face, args.glasses, args.output)
    
    if result_path:
        print(f"Try-on result saved to {result_path}")
    else:
        print("Failed to perform virtual try-on")
