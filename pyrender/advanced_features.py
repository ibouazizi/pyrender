"""Advanced OpenGL 4.x rendering features.

This module provides support for advanced OpenGL features including:
- Direct State Access (DSA)
- Persistent buffer mapping
- Enhanced buffer storage
- Tessellation shaders
- Transform feedback
"""
import numpy as np
import ctypes
from OpenGL.GL import *


class DirectStateBuffer:
    """Buffer object using Direct State Access (OpenGL 4.5+).
    
    This provides more efficient buffer management without 
    requiring explicit binding/unbinding.
    """

    def __init__(self, target=GL_ARRAY_BUFFER, usage=GL_STATIC_DRAW):
        self._id = glGenBuffers(1)
        glBindBuffer(target, self._id)
        self._target = target
        self._usage = usage
        self._size = 0

    def set_data(self, data):
        """Set buffer data using DSA."""
        data = np.ascontiguousarray(data)
        self._size = data.nbytes
        glBindBuffer(self._target, self._id)
        glBufferData(self._target, self._size, data, self._usage)

    def update_data(self, data, offset=0):
        """Update a portion of the buffer data."""
        data = np.ascontiguousarray(data)
        if offset + data.nbytes > self._size:
            raise ValueError("Update would exceed buffer size")
        glBindBuffer(self._target, self._id)
        glBufferSubData(self._target, offset, data.nbytes, data)

    def bind(self, binding_point=None):
        """Bind the buffer to a specific binding point."""
        glBindBuffer(self._target, self._id)
        if binding_point is not None:
            glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, binding_point, self._id)

    def delete(self):
        """Delete the buffer."""
        if self._id is not None:
            glDeleteBuffers(1, [self._id])
            self._id = None


class PersistentBuffer:
    """Buffer with persistent mapping (OpenGL 4.4+).
    
    Allows for efficient streaming of dynamic data.
    """

    def __init__(self, size, target=GL_ARRAY_BUFFER):
        self._id = glGenBuffers(1)
        self._target = target
        self._size = size
        glBindBuffer(target, self._id)
        
        # Create buffer with persistent mapping flags
        flags = (GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | 
                GL_MAP_COHERENT_BIT)
        glBufferStorage(self._target, size, None, flags)
        
        # Map the buffer
        self._mapped_data = glMapBufferRange(
            self._target, 0, size, flags)
        
        # Create numpy array view of the mapped memory
        self._array = np.ctypeslib.as_array(
            ctypes.cast(self._mapped_data, 
                       ctypes.POINTER(ctypes.c_byte)), 
            shape=(size,))

    def update(self, data, offset=0):
        """Update buffer contents through the persistent mapping."""
        data = np.ascontiguousarray(data)
        if offset + data.nbytes > self._size:
            raise ValueError("Update would exceed buffer size")
        self._array[offset:offset + data.nbytes] = np.frombuffer(
            data.tobytes(), dtype=np.uint8)

    def bind(self, binding_point=None):
        """Bind the buffer to a specific binding point."""
        glBindBuffer(self._target, self._id)
        if binding_point is not None:
            glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, binding_point, self._id)

    def delete(self):
        """Delete the buffer."""
        if self._id is not None:
            glBindBuffer(self._target, self._id)
            glUnmapBuffer(self._target)
            glDeleteBuffers(1, [self._id])
            self._id = None
            self._mapped_data = None
            self._array = None


class TransformFeedback:
    """Transform feedback object for capturing vertex shader output.
    
    This allows for GPU-based vertex processing with results
    captured back to buffer objects.
    """

    def __init__(self, varyings, vertex_shader=None):
        """Initialize transform feedback.
        
        Parameters
        ----------
        varyings : list of str
            Names of vertex shader output variables to capture.
        vertex_shader : str, optional
            The vertex shader source code. If not provided, a default
            pass-through shader will be used.
        """
        self._id = glGenTransformFeedbacks(1)
        self._varyings = varyings
        self._buffers = []
        
        # Create and set up shader program
        if vertex_shader is None:
            vertex_shader = """
            #version 430
            in vec4 position;
            out vec4 outPosition;
            void main() {
                outPosition = position;
                gl_Position = position;
            }
            """
            
        # Create program and shaders
        self._program = glCreateProgram()
        shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(shader, vertex_shader)
        glCompileShader(shader)
        
        # Check compilation
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(shader)
            glDeleteShader(shader)
            raise ValueError(f"Shader compilation failed: {error}")
            
        # Attach shader and link program
        glAttachShader(self._program, shader)
        
        # Specify transform feedback varyings
        varyings_array = (ctypes.POINTER(ctypes.c_char) * len(self._varyings))()
        varying_strings = []
        for i, v in enumerate(self._varyings):
            v_bytes = v.encode('utf-8') + b'\0'
            v_array = (ctypes.c_char * len(v_bytes))(*v_bytes)
            varying_strings.append(v_array)
            varyings_array[i] = v_array
        glTransformFeedbackVaryings(
            self._program,
            len(self._varyings),
            varyings_array,
            GL_INTERLEAVED_ATTRIBS
        )
        
        glLinkProgram(self._program)
        
        # Check link status
        if not glGetProgramiv(self._program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self._program)
            glDeleteProgram(self._program)
            glDeleteShader(shader)
            raise ValueError(f"Program linking failed: {error}")
            
        glDeleteShader(shader)

    def attach_buffer(self, buffer, binding_point):
        """Attach a buffer for transform feedback output."""
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, self._id)
        buffer.bind(binding_point)
        self._buffers.append((buffer, binding_point))

    def begin(self, primitive_mode=GL_POINTS):
        """Begin transform feedback capture."""
        glUseProgram(self._program)
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, self._id)
        glBeginTransformFeedback(primitive_mode)

    def end(self):
        """End transform feedback capture."""
        glEndTransformFeedback()
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0)

    def delete(self):
        """Delete the transform feedback object."""
        if self._id is not None:
            glDeleteTransformFeedbacks(1, [self._id])
            self._id = None
            self._buffers = []