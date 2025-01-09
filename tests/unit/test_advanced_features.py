"""Unit tests for advanced OpenGL features.
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Use EGL for headless testing

import numpy as np
import pytest
import pyrender
from pyrender.advanced_features import (
    DirectStateBuffer,
    PersistentBuffer,
    TransformFeedback
)
from pyrender.platforms.egl import EGLPlatform

@pytest.fixture(scope='module')
def gl_context():
    """Create an OpenGL context for testing."""
    platform = EGLPlatform(640, 480)
    platform.init_context()
    platform.make_current()
    yield platform
    platform.delete_context()

def test_direct_state_buffer(gl_context):
    """Test DirectStateBuffer functionality."""
    # Create buffer
    buffer = DirectStateBuffer()
    
    # Set data
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buffer.set_data(data)
    
    # Update data
    new_data = np.array([5.0, 6.0], dtype=np.float32)
    buffer.update_data(new_data, offset=0)
    
    # Clean up
    buffer.delete()
    assert buffer._id is None

def test_persistent_buffer(gl_context):
    """Test PersistentBuffer functionality."""
    # Create buffer
    size = 1024  # bytes
    buffer = PersistentBuffer(size)
    
    # Update data
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buffer.update(data)
    
    # Clean up
    buffer.delete()
    assert buffer._id is None
    assert buffer._mapped_data is None
    assert buffer._array is None

def test_transform_feedback(gl_context):
    """Test TransformFeedback functionality."""
    # Create transform feedback
    vertex_shader = """
    #version 430
    in vec4 position;
    in vec4 velocity;
    out vec4 outPosition;
    out vec4 outVelocity;
    void main() {
        outPosition = position;
        outVelocity = velocity;
        gl_Position = position;
    }
    """
    varyings = ['outPosition', 'outVelocity']
    tf = TransformFeedback(varyings, vertex_shader)
    
    # Create and attach buffer
    buffer = DirectStateBuffer()
    data = np.zeros(100, dtype=np.float32)
    buffer.set_data(data)
    
    tf.attach_buffer(buffer, 0)
    
    # Test capture
    tf.begin()
    tf.end()
    
    # Clean up
    tf.delete()
    assert tf._id is None
    assert len(tf._buffers) == 0

def test_buffer_binding_points(gl_context):
    """Test buffer binding points don't conflict."""
    buffer1 = DirectStateBuffer()
    buffer2 = DirectStateBuffer()
    
    data1 = np.array([1.0, 2.0], dtype=np.float32)
    data2 = np.array([3.0, 4.0], dtype=np.float32)
    
    buffer1.set_data(data1)
    buffer2.set_data(data2)
    
    # Bind to different points
    buffer1.bind(0)
    buffer2.bind(1)
    
    # Clean up
    buffer1.delete()
    buffer2.delete()