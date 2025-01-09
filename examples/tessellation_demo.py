"""Example of using tessellation shaders for displacement mapping.
"""
import numpy as np
import trimesh
import pyrender
from PIL import Image

def create_displacement_material():
    """Create a material that uses tessellation shaders for displacement mapping."""
    # Load shader sources
    with open('pyrender/shaders/tessellation/displacement.tesc', 'r') as f:
        tesc_source = f.read()
    with open('pyrender/shaders/tessellation/displacement.tese', 'r') as f:
        tese_source = f.read()
        
    # Create height map texture
    height_map = np.random.uniform(0, 1, (512, 512)).astype(np.float32)
    height_map = Image.fromarray((height_map * 255).astype(np.uint8))
    
    # Create material with tessellation shaders
    material = pyrender.MetallicRoughnessMaterial(
        tess_control_shader=tesc_source,
        tess_eval_shader=tese_source,
        height_map=height_map,
        roughnessFactor=0.7,
        metallicFactor=0.2,
    )
    
    return material

def main():
    # Create scene
    scene = pyrender.Scene()
    
    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    c_pose = np.eye(4)
    c_pose[2, 3] = 2.0
    scene.add(camera, pose=c_pose)
    
    # Create plane mesh with high tessellation
    plane = trimesh.creation.box(extents=[2, 2, 0.1])
    material = create_displacement_material()
    
    # Set tessellation parameters
    tess_params = {
        'tessLevel': 16.0,
        'tessMultiplier': 1.0,
        'maxTessLevel': 64.0,
        'cameraPosition': c_pose[:3, 3],
    }
    
    # Create mesh with tessellation
    mesh = pyrender.Mesh.from_trimesh(
        plane,
        material=material,
        smooth=True,
        tess_params=tess_params
    )
    
    # Add mesh to scene
    scene.add(mesh)
    
    # Add lighting
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=c_pose)
    
    # Show the scene
    pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == '__main__':
    main()