"""Unit tests for compute shader functionality.
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Use EGL for headless testing

import numpy as np
import pytest
import pyrender
from pyrender.compute_shader import ComputeShader
from pyrender.platforms.egl import EGLPlatform

@pytest.fixture(scope='module')
def gl_context():
    """Create an OpenGL context for testing."""
    platform = EGLPlatform(640, 480)
    platform.init_context()
    platform.make_current()
    yield platform
    platform.delete_context()

def test_compute_shader_creation(gl_context):
    """Test creating a compute shader."""
    source = """
    #version 430
    layout(local_size_x = 1) in;
    layout(std430, binding = 0) buffer Data { float data[]; };
    void main() {
        uint idx = gl_GlobalInvocationID.x;
        data[idx] = data[idx] * 2.0;
    }
    """
    
    shader = ComputeShader(source, work_groups=(1, 1, 1))
    assert shader._program_id is not None

def test_compute_shader_buffer(gl_context):
    """Test compute shader buffer operations."""
    source = """
    #version 430
    layout(local_size_x = 1) in;
    layout(std430, binding = 0) buffer Data { float data[]; };
    void main() {
        uint idx = gl_GlobalInvocationID.x;
        data[idx] = data[idx] * 2.0;
    }
    """
    
    shader = ComputeShader(source, work_groups=(4, 1, 1))
    
    # Create and set buffer data
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    shader.create_storage_buffer('data', data, 0)
    
    # Run compute shader
    shader.dispatch()
    
    # Read back results
    result = shader.read_buffer('data')
    expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)

def test_compute_shader_multiple_buffers(gl_context):
    """Test compute shader with multiple buffers."""
    source = """
    #version 430
    layout(local_size_x = 1) in;
    layout(std430, binding = 0) buffer InputData { float input_data[]; };
    layout(std430, binding = 1) buffer OutputData { float output_data[]; };
    void main() {
        uint idx = gl_GlobalInvocationID.x;
        output_data[idx] = input_data[idx] * 2.0;
    }
    """
    
    shader = ComputeShader(source, work_groups=(4, 1, 1))
    
    # Create input and output buffers
    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    output_data = np.zeros(4, dtype=np.float32)
    
    shader.create_storage_buffer('input', input_data, 0)
    shader.create_storage_buffer('output', output_data, 1)
    
    # Run compute shader
    shader.dispatch()
    
    # Read back results
    result = shader.read_buffer('output')
    expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)

def test_compute_shader_cleanup(gl_context):
    """Test proper cleanup of compute shader resources."""
    source = """
    #version 430
    layout(local_size_x = 1) in;
    void main() {}
    """
    
    shader = ComputeShader(source, work_groups=(1, 1, 1))
    shader.delete()
    assert shader._program_id is None
    assert len(shader._ssbo_map) == 0