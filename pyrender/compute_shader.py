"""OpenGL compute shader support for advanced rendering features.

This module adds support for OpenGL 4.3+ compute shaders, enabling
parallel processing capabilities for tasks like particle systems,
physics simulations, and post-processing effects.
"""
import numpy as np
from OpenGL.GL import *

class ComputeShader:
    """A wrapper for OpenGL compute shaders.

    Parameters
    ----------
    source : str
        The compute shader source code.
    work_groups : tuple of int
        The number of work groups in (x, y, z) dimensions.
    """

    def __init__(self, source, work_groups=(1, 1, 1)):
        self._program_id = None
        self._source = source
        self._work_groups = work_groups
        self._ssbo_map = {}  # Maps buffer names to buffer IDs
        self._add_to_context()

    def _add_to_context(self):
        """Compile and link the compute shader program."""
        if self._program_id is not None:
            raise ValueError('Compute shader already in context')

        # Compile compute shader
        shader = glCreateShader(GL_COMPUTE_SHADER)
        glShaderSource(shader, self._source)
        glCompileShader(shader)

        # Check compilation status
        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        error = glGetShaderInfoLog(shader)
        print(f"Shader compilation status: {status}")
        print(f"Shader info log: {error}")
        if not status:
            glDeleteShader(shader)
            raise ValueError(f'Compute shader compilation failed: {error}\nSource:\n{self._source}')

        # Create and link program
        self._program_id = glCreateProgram()
        print(f"Created program with ID: {self._program_id}")
        glAttachShader(self._program_id, shader)
        glLinkProgram(self._program_id)

        # Check link status
        status = glGetProgramiv(self._program_id, GL_LINK_STATUS)
        error = glGetProgramInfoLog(self._program_id)
        print(f"Program link status: {status}")
        print(f"Program info log: {error}")
        if not status:
            glDeleteProgram(self._program_id)
            glDeleteShader(shader)
            raise ValueError(f'Compute shader program linking failed: {error}\nSource:\n{self._source}')

        glDeleteShader(shader)

    def create_storage_buffer(self, name, data, binding):
        """Create a shader storage buffer object (SSBO).

        Parameters
        ----------
        name : str
            Name of the buffer (must match the name in the shader).
        data : numpy.ndarray
            The data to store in the buffer.
        binding : int
            The binding point for the buffer.
        """
        if name in self._ssbo_map:
            raise ValueError(f'Buffer {name} already exists')

        # Create and bind buffer
        ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        
        # Initialize buffer with data
        data_ptr = data.ctypes.data_as(ctypes.c_void_p)
        glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data_ptr, GL_DYNAMIC_COPY)
        
        # Bind to shader
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ssbo)
        
        self._ssbo_map[name] = (ssbo, binding, data.dtype, data.shape)

    def update_buffer(self, name, data):
        """Update the contents of a storage buffer.

        Parameters
        ----------
        name : str
            Name of the buffer to update.
        data : numpy.ndarray
            New data for the buffer.
        """
        if name not in self._ssbo_map:
            raise ValueError(f'Buffer {name} does not exist')

        ssbo, binding, dtype, shape = self._ssbo_map[name]
        if data.dtype != dtype or data.shape != shape:
            raise ValueError('Data type or shape mismatch')

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        data_ptr = data.ctypes.data_as(ctypes.c_void_p)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, data.nbytes, data_ptr)

    def read_buffer(self, name):
        """Read data back from a storage buffer.

        Parameters
        ----------
        name : str
            Name of the buffer to read.

        Returns
        -------
        numpy.ndarray
            The buffer data.
        """
        if name not in self._ssbo_map:
            raise ValueError(f'Buffer {name} does not exist')

        ssbo, binding, dtype, shape = self._ssbo_map[name]
        size = np.prod(shape) * dtype.itemsize

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        data_ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY)
        data = np.ctypeslib.as_array(ctypes.cast(data_ptr, 
                                                ctypes.POINTER(ctypes.c_byte)), 
                                    shape=(size,))
        data = data.view(dtype=dtype).reshape(shape)
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)

        return data.copy()

    def dispatch(self):
        """Execute the compute shader."""
        print(f"Using program with ID: {self._program_id}")
        glUseProgram(self._program_id)
        glDispatchCompute(*self._work_groups)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def delete(self):
        """Delete the compute shader program and associated buffers."""
        if self._program_id is not None:
            glDeleteProgram(self._program_id)
            self._program_id = None

        for ssbo, _, _, _ in self._ssbo_map.values():
            glDeleteBuffers(1, [ssbo])
        self._ssbo_map.clear()

    def __del__(self):
        self.delete()