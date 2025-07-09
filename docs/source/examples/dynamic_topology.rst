Dynamic Topology Updates
========================

PyRender now supports dynamic topology updates, allowing you to change mesh connectivity
without recreating the entire mesh object. This is particularly useful for applications
like skeleton visualization where bone connections may change during animation.

Key Features
------------

1. **Dynamic Index Buffer Updates**: Change mesh connectivity on the fly
2. **Dynamic Vertex Position Updates**: Update vertex positions without recreation
3. **Optimized GPU Buffer Management**: Pre-allocate buffers with reserve space
4. **Scene Graph Integration**: Automatic bounds invalidation

Basic Usage
-----------

Updating Primitive Topology
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    import pyrender

    # Create a primitive with initial connectivity
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    
    initial_indices = np.array([[0, 1], [1, 3], [3, 2], [2, 0]])
    
    primitive = pyrender.Primitive(
        positions=positions,
        indices=initial_indices,
        mode=pyrender.constants.GLTF.LINES
    )
    
    # Later, update the connectivity
    new_indices = np.array([[0, 3], [1, 2]])
    primitive.update_topology(new_indices)

Updating Through Mesh Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create a mesh
    mesh = pyrender.Mesh(primitives=[primitive])
    
    # Update topology of the first primitive
    mesh.update_topology(new_indices, primitive_index=0)
    
    # Update positions
    new_positions = positions + np.array([0, 0, 1])
    mesh.update_positions(new_positions, primitive_index=0)

Buffer Pre-allocation
^^^^^^^^^^^^^^^^^^^^^

For better performance when frequently updating topology, you can pre-allocate
larger buffers:

.. code-block:: python

    # Reserve 2x the initial buffer size
    primitive = pyrender.Primitive(
        positions=positions,
        indices=initial_indices,
        mode=pyrender.constants.GLTF.LINES,
        buffer_reserve_ratio=2.0
    )

This avoids GPU buffer reallocation when the topology grows within the reserved space.

Skeleton Visualization Example
------------------------------

Here's a complete example showing dynamic skeleton visualization:

.. code-block:: python

    import numpy as np
    import pyrender
    
    def create_skeleton_primitive(joint_positions, bone_connections):
        """Create a line primitive for skeleton bones."""
        return pyrender.Primitive(
            positions=joint_positions,
            indices=bone_connections,
            mode=pyrender.constants.GLTF.LINES,
            buffer_reserve_ratio=2.0  # Reserve space for dynamic updates
        )
    
    # Initial skeleton
    joints = np.array([
        [0, 0, 0],      # Root
        [0, 1, 0],      # Spine
        [-0.5, 1.5, 0], # Left shoulder
        [0.5, 1.5, 0],  # Right shoulder
    ])
    
    # Initial connections
    bones = np.array([[0, 1], [1, 2], [1, 3]])
    
    # Create scene
    scene = pyrender.Scene()
    skeleton_prim = create_skeleton_primitive(joints, bones)
    skeleton_mesh = pyrender.Mesh(primitives=[skeleton_prim])
    scene.add(skeleton_mesh)
    
    # During animation, update connections
    new_bones = np.array([[0, 1], [1, 2], [1, 3], [2, 3]])  # Add shoulder connection
    skeleton_prim.update_topology(new_bones)
    
    # Don't forget to invalidate scene bounds
    scene.invalidate_bounds()

Performance Considerations
--------------------------

1. **Use Buffer Reserve Ratio**: Pre-allocate buffers when you know the maximum size
2. **Batch Updates**: Update multiple attributes together when possible
3. **Invalidate Bounds**: Call ``scene.invalidate_bounds()`` after geometry changes
4. **Minimize Reallocations**: The system uses ``glBufferSubData`` when possible

API Reference
-------------

Primitive Methods
^^^^^^^^^^^^^^^^^

.. method:: Primitive.update_topology(indices)

    Update the index buffer dynamically.
    
    :param indices: New face indices, or None for non-indexed primitives
    :type indices: array_like or None

.. method:: Primitive.update_positions(positions)

    Update vertex positions dynamically.
    
    :param positions: New vertex positions
    :type positions: (n,3) array_like

Mesh Methods
^^^^^^^^^^^^

.. method:: Mesh.update_topology(indices, primitive_index=0)

    Update topology of a specific primitive in the mesh.
    
    :param indices: New face indices
    :type indices: array_like or None
    :param primitive_index: Index of primitive to update
    :type primitive_index: int

.. method:: Mesh.update_positions(positions, primitive_index=0)

    Update positions of a specific primitive in the mesh.
    
    :param positions: New vertex positions
    :type positions: (n,3) array_like
    :param primitive_index: Index of primitive to update
    :type primitive_index: int

Scene Methods
^^^^^^^^^^^^^

.. method:: Scene.invalidate_bounds()

    Invalidate cached scene bounds. Call this after updating mesh geometry.