#version 410

// Triangles with clockwise vertex order
layout(triangles, equal_spacing, ccw) in;

// Input from tessellation control shader
in vec3 tcPosition[];
in vec2 tcTexCoord[];
in vec3 tcNormal[];

// Output to fragment shader
out vec3 tePosition;
out vec2 teTexCoord;
out vec3 teNormal;

// Displacement parameters
layout(std140) uniform DisplacementParams {
    mat4 modelViewMatrix;
    mat4 projectionMatrix;
    float displacementScale;
    float displacementBias;
};

// Height map texture
uniform sampler2D heightMap;

// Interpolate position, texcoord, and normal using barycentric coordinates
vec3 interpolate3D(vec3 v0, vec3 v1, vec3 v2) {
    return gl_TessCoord.x * v0 + gl_TessCoord.y * v1 + gl_TessCoord.z * v2;
}

vec2 interpolate2D(vec2 v0, vec2 v1, vec2 v2) {
    return gl_TessCoord.x * v0 + gl_TessCoord.y * v1 + gl_TessCoord.z * v2;
}

void main() {
    // Interpolate position, texcoord, and normal
    vec3 position = interpolate3D(tcPosition[0], tcPosition[1], tcPosition[2]);
    vec2 texCoord = interpolate2D(tcTexCoord[0], tcTexCoord[1], tcTexCoord[2]);
    vec3 normal = normalize(interpolate3D(tcNormal[0], tcNormal[1], tcNormal[2]));

    // Sample height map and displace vertex along normal
    float height = texture(heightMap, texCoord).r;
    float displacement = height * displacementScale + displacementBias;
    position += normal * displacement;

    // Transform position to clip space
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);

    // Pass interpolated attributes to fragment shader
    tePosition = position;
    teTexCoord = texCoord;
    teNormal = normal;
}