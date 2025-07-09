"""Demo of dynamic topology updates for skeleton visualization.

This example shows how to use the new update_topology() method to dynamically
change mesh connectivity without recreating the entire mesh.
"""

import numpy as np
import pyrender
import trimesh

def create_skeleton_mesh(joint_positions, connections):
    """Create a mesh representing a skeleton with joints and bones.
    
    Parameters
    ----------
    joint_positions : (n, 3) array
        3D positions of joints
    connections : (m, 2) array
        Pairs of joint indices that are connected
    
    Returns
    -------
    mesh : pyrender.Mesh
        A mesh with spheres for joints and lines for bones
    """
    # Create spheres for joints
    spheres = []
    for pos in joint_positions:
        sphere = trimesh.creation.uv_sphere(radius=0.05, count=[8, 8])
        sphere.apply_translation(pos)
        spheres.append(sphere)
    
    joint_mesh = trimesh.util.concatenate(spheres)
    
    # Create line primitive for bones
    bone_positions = joint_positions
    bone_indices = connections
    
    # Create pyrender meshes
    joint_prim = pyrender.Primitive(
        positions=joint_mesh.vertices,
        indices=joint_mesh.faces,
        mode=pyrender.constants.GLTF.TRIANGLES,
    )
    
    bone_prim = pyrender.Primitive(
        positions=bone_positions,
        indices=bone_indices,
        mode=pyrender.constants.GLTF.LINES,
        buffer_reserve_ratio=2.0  # Reserve extra space for dynamic updates
    )
    
    # Create mesh with both primitives
    mesh = pyrender.Mesh(primitives=[joint_prim, bone_prim])
    
    return mesh, bone_prim

def update_skeleton_connections(bone_primitive, new_connections):
    """Update the bone connections dynamically.
    
    Parameters
    ----------
    bone_primitive : pyrender.Primitive
        The primitive representing bones
    new_connections : (m, 2) array
        New pairs of joint indices that are connected
    """
    bone_primitive.update_topology(new_connections)

def main():
    # Initial skeleton configuration
    initial_joints = np.array([
        [0.0, 0.0, 0.0],    # Root
        [0.0, 1.0, 0.0],    # Spine
        [-0.5, 1.5, 0.0],   # Left shoulder
        [0.5, 1.5, 0.0],    # Right shoulder
        [0.0, 2.0, 0.0],    # Head
        [-0.5, 0.5, 0.0],   # Left elbow
        [0.5, 0.5, 0.0],    # Right elbow
    ])
    
    # Initial connections (simple skeleton)
    initial_connections = np.array([
        [0, 1],  # Root to spine
        [1, 2],  # Spine to left shoulder
        [1, 3],  # Spine to right shoulder
        [1, 4],  # Spine to head
    ])
    
    # Create scene
    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2])
    
    # Create skeleton mesh
    skeleton_mesh, bone_prim = create_skeleton_mesh(initial_joints, initial_connections)
    scene.add(skeleton_mesh)
    
    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 4.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=cam_pose)
    
    # Add light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=cam_pose)
    
    # Create viewer
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, 
                           run_in_thread=True, record=False)
    
    # Demonstrate dynamic topology updates
    import time
    
    # Configuration 1: Add arm connections
    config1_connections = np.array([
        [0, 1],  # Root to spine
        [1, 2],  # Spine to left shoulder
        [1, 3],  # Spine to right shoulder
        [1, 4],  # Spine to head
        [2, 5],  # Left shoulder to left elbow
        [3, 6],  # Right shoulder to right elbow
    ])
    
    # Configuration 2: Cross connections
    config2_connections = np.array([
        [0, 1],  # Root to spine
        [1, 2],  # Spine to left shoulder
        [1, 3],  # Spine to right shoulder
        [1, 4],  # Spine to head
        [2, 6],  # Left shoulder to right elbow (cross)
        [3, 5],  # Right shoulder to left elbow (cross)
    ])
    
    # Configuration 3: Star pattern from spine
    config3_connections = np.array([
        [1, 0],  # Spine to root
        [1, 2],  # Spine to left shoulder
        [1, 3],  # Spine to right shoulder
        [1, 4],  # Spine to head
        [1, 5],  # Spine to left elbow
        [1, 6],  # Spine to right elbow
    ])
    
    print("Demonstrating dynamic topology updates...")
    print("Press Ctrl+C to exit")
    
    configs = [initial_connections, config1_connections, config2_connections, config3_connections]
    config_names = ["Initial", "Arms Added", "Cross Pattern", "Star Pattern"]
    
    try:
        config_idx = 0
        while viewer.is_active:
            time.sleep(2.0)
            
            # Cycle through configurations
            config_idx = (config_idx + 1) % len(configs)
            new_connections = configs[config_idx]
            
            print(f"Switching to configuration: {config_names[config_idx]}")
            
            # Update topology dynamically
            update_skeleton_connections(bone_prim, new_connections)
            
            # Invalidate scene bounds to ensure proper rendering
            scene.invalidate_bounds()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    
    viewer.close_external()

if __name__ == "__main__":
    main()