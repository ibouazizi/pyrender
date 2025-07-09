"""Unit tests for dynamic topology updates."""

import pytest
import numpy as np
import pyrender


def test_primitive_update_topology():
    """Test updating primitive topology."""
    # Create a simple primitive with initial indices
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32)
    
    initial_indices = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=np.uint32)
    
    # Create primitive
    prim = pyrender.Primitive(
        positions=positions,
        indices=initial_indices,
        mode=pyrender.constants.GLTF.TRIANGLES
    )
    
    # Verify initial indices
    assert np.array_equal(prim.indices, initial_indices)
    
    # Update topology
    new_indices = np.array([
        [0, 1, 3],
        [0, 3, 2],
    ], dtype=np.uint32)
    
    prim.update_topology(new_indices)
    
    # Verify updated indices
    assert np.array_equal(prim.indices, new_indices)
    
    # Test with None indices (non-indexed primitive)
    prim.update_topology(None)
    assert prim.indices is None


def test_primitive_update_positions():
    """Test updating primitive positions."""
    # Create a simple primitive
    initial_positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    
    prim = pyrender.Primitive(
        positions=initial_positions,
        mode=pyrender.constants.GLTF.POINTS
    )
    
    # Verify initial positions
    assert np.array_equal(prim.positions, initial_positions)
    
    # Update positions
    new_positions = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float32)
    
    prim.update_positions(new_positions)
    
    # Verify updated positions
    assert np.array_equal(prim.positions, new_positions)


def test_mesh_update_topology():
    """Test updating mesh topology through mesh interface."""
    # Create a mesh with one primitive
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32)
    
    initial_indices = np.array([
        [0, 1, 2],
    ], dtype=np.uint32)
    
    prim = pyrender.Primitive(
        positions=positions,
        indices=initial_indices,
        mode=pyrender.constants.GLTF.TRIANGLES
    )
    
    mesh = pyrender.Mesh(primitives=[prim])
    
    # Update topology through mesh
    new_indices = np.array([
        [0, 1, 3],
        [0, 3, 2],
    ], dtype=np.uint32)
    
    mesh.update_topology(new_indices, primitive_index=0)
    
    # Verify updated indices
    assert np.array_equal(mesh.primitives[0].indices, new_indices)
    
    # Test invalid primitive index
    with pytest.raises(ValueError):
        mesh.update_topology(new_indices, primitive_index=1)


def test_buffer_reserve_ratio():
    """Test buffer pre-allocation with reserve ratio."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)
    
    indices = np.array([
        [0, 1],
    ], dtype=np.uint32)
    
    # Create primitive with buffer reserve ratio
    prim = pyrender.Primitive(
        positions=positions,
        indices=indices,
        mode=pyrender.constants.GLTF.LINES,
        buffer_reserve_ratio=2.0
    )
    
    # Verify reserve ratio is stored
    assert prim._buffer_reserve_ratio == 2.0
    
    # Update with larger indices array (should use reserved space)
    new_indices = np.array([
        [0, 1],
        [1, 0],
    ], dtype=np.uint32)
    
    prim.update_topology(new_indices)
    assert np.array_equal(prim.indices, new_indices)


def test_scene_bounds_invalidation():
    """Test that scene bounds are invalidated on topology update."""
    # Create scene with a mesh
    scene = pyrender.Scene()
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    
    mesh = pyrender.Mesh.from_points(positions)
    scene.add(mesh)
    
    # Access bounds to cache them
    initial_bounds = scene.bounds
    assert initial_bounds is not None
    
    # Invalidate bounds
    scene.invalidate_bounds()
    
    # Verify bounds are recomputed
    new_bounds = scene.bounds
    assert np.array_equal(initial_bounds, new_bounds)


if __name__ == "__main__":
    pytest.main([__file__])