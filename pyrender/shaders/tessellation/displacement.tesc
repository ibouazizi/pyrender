#version 410

// Number of control points in output patch
layout(vertices = 3) out;

// Input attributes from vertex shader
in vec3 vPosition[];
in vec2 vTexCoord[];
in vec3 vNormal[];

// Output to tessellation evaluation shader
out vec3 tcPosition[];
out vec2 tcTexCoord[];
out vec3 tcNormal[];

// Tessellation parameters
layout(std140) uniform TessParams {
    float tessLevel;          // Base tessellation level
    float tessMultiplier;     // Distance-based multiplier
    float maxTessLevel;       // Maximum tessellation level
    vec3 cameraPosition;      // Camera position for LOD
};

float getLODLevel(vec3 pos) {
    float distance = distance(pos, cameraPosition);
    return clamp(tessLevel * tessMultiplier / distance, 1.0, maxTessLevel);
}

void main() {
    // Pass through vertex attributes
    tcPosition[gl_InvocationID] = vPosition[gl_InvocationID];
    tcTexCoord[gl_InvocationID] = vTexCoord[gl_InvocationID];
    tcNormal[gl_InvocationID] = vNormal[gl_InvocationID];

    // Calculate tessellation levels based on distance from camera
    if (gl_InvocationID == 0) {
        vec3 center = (vPosition[0] + vPosition[1] + vPosition[2]) / 3.0;
        float lodLevel = getLODLevel(center);

        // Set tessellation levels for edges and interior
        gl_TessLevelOuter[0] = lodLevel;
        gl_TessLevelOuter[1] = lodLevel;
        gl_TessLevelOuter[2] = lodLevel;
        gl_TessLevelInner[0] = lodLevel;
    }
}