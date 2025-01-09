"""Example of using compute shaders for particle systems.
"""
import numpy as np
import trimesh
import time
import pyrender
from pyrender.compute_shader import ComputeShader
from pyrender.advanced_features import PersistentBuffer

# Create particle system data
N_PARTICLES = 100000

def create_particle_data():
    # Initialize random particle data
    particles = np.zeros(N_PARTICLES, dtype=[
        ('position', 'f4', 4),  # xyz = position, w = life
        ('velocity', 'f4', 4),  # xyz = velocity, w = size
        ('color', 'f4', 4),     # rgba color
    ])
    
    # Random positions in a sphere
    theta = np.random.uniform(0, 2*np.pi, N_PARTICLES)
    phi = np.arccos(2 * np.random.uniform(0, 1, N_PARTICLES) - 1)
    r = np.power(np.random.uniform(0, 1, N_PARTICLES), 1/3) * 2.0
    
    particles['position'][:, 0] = r * np.sin(phi) * np.cos(theta)
    particles['position'][:, 1] = r * np.sin(phi) * np.sin(theta)
    particles['position'][:, 2] = r * np.cos(phi)
    particles['position'][:, 3] = np.random.uniform(0, 1, N_PARTICLES)  # life
    
    # Random velocities
    particles['velocity'][:, :3] = np.random.uniform(-1, 1, (N_PARTICLES, 3))
    particles['velocity'][:, 3] = np.random.uniform(0.1, 0.3, N_PARTICLES)  # size
    
    # Random colors
    particles['color'] = np.random.uniform(0.5, 1.0, (N_PARTICLES, 4))
    particles['color'][:, 3] = 1.0  # alpha
    
    return particles

def main():
    # Set up EGL context
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    from pyrender.platforms.egl import EGLPlatform
    platform = EGLPlatform(640, 480)
    platform.init_context()
    platform.make_current()

    # Initialize scene
    scene = pyrender.Scene()
    
    # Create camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 5.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    # Add lighting
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)
    
    # Create initial particle data
    particles = create_particle_data()
    
    # Create point cloud mesh
    positions = particles['position'][:, :3]
    colors = particles['color']
    cloud = pyrender.Mesh.from_points(positions, colors=colors)
    
    # Add to scene
    main.cloud_node = scene.add(cloud)
    
    # Create offscreen renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    
    # Main loop
    for i in range(10):  # Run for 10 frames
        # Update particle positions
        dt = 0.016
        gravity = np.array([0.0, -9.81, 0.0])
        attractor_pos = np.array([0.0, 5.0 * np.sin(time.time()), 0.0])
        attractor_strength = 10.0
        
        # Update particles
        for j in range(len(particles)):
            # Update life
            particles[j]['position'][3] -= dt
            
            # Reset dead particles
            if particles[j]['position'][3] <= 0.0:
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.arccos(2 * np.random.uniform(0, 1) - 1)
                r = np.power(np.random.uniform(0, 1), 1/3) * 2.0
                
                particles[j]['position'][:3] = [
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ]
                particles[j]['position'][3] = 1.0 + np.random.uniform(0, 1)
                
                particles[j]['velocity'][:3] = np.random.uniform(-1, 1, 3)
                particles[j]['velocity'][3] = 0.1 + np.random.uniform(0, 0.2)
                
                particles[j]['color'] = np.array([
                    0.5 + 0.5 * np.random.uniform(0, 1),
                    0.5 + 0.5 * np.random.uniform(0, 1),
                    0.5 + 0.5 * np.random.uniform(0, 1),
                    1.0
                ])
            else:
                # Apply forces
                pos = particles[j]['position'][:3]
                to_attractor = attractor_pos - pos
                dist = np.linalg.norm(to_attractor)
                force = to_attractor / dist * attractor_strength / (dist * dist + 1.0)
                
                # Update velocity and position
                particles[j]['velocity'][:3] += (force + gravity) * dt
                particles[j]['position'][:3] += particles[j]['velocity'][:3] * dt
                
                # Fade out color
                particles[j]['color'][3] = particles[j]['position'][3]
        
        # Update mesh with new particle positions
        positions = particles['position'][:, :3]
        colors = particles['color']
        
        # Create point cloud mesh
        cloud = pyrender.Mesh.from_points(positions, colors=colors)
        
        # Update scene
        scene.remove_node(main.cloud_node)
        main.cloud_node = scene.add(cloud)
        
        # Render frame
        color, depth = renderer.render(scene)
        
        # Save frame
        import imageio
        imageio.imwrite(f'frame_{i:03d}.png', color)

if __name__ == '__main__':
    main()