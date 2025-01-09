#version 410

// Input attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 velocity;
layout(location = 2) in float life;
layout(location = 3) in float size;

// Output varyings for transform feedback
out vec3 outPosition;
out vec3 outVelocity;
out float outLife;
out float outSize;

// Simulation parameters
layout(std140) uniform SimParams {
    float deltaTime;
    vec3 gravity;
    vec3 wind;
    float damping;
    float minLife;
    float maxLife;
};

// Random number generation
float rand(vec2 co) {
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main() {
    // Update life
    outLife = life - deltaTime;
    
    // Reset dead particles
    if (outLife <= 0.0) {
        // Random position in emission volume
        float theta = rand(position.xy) * 2.0 * 3.14159;
        float phi = acos(2.0 * rand(position.yz) - 1.0);
        float r = pow(rand(position.xz), 0.33333) * 2.0;
        
        outPosition = vec3(
            r * sin(phi) * cos(theta),
            r * sin(phi) * sin(theta),
            r * cos(phi)
        );
        
        // Random initial velocity
        outVelocity = normalize(outPosition) * (2.0 + rand(velocity.xy));
        outLife = minLife + rand(vec2(life, size)) * (maxLife - minLife);
        outSize = 0.1 + rand(vec2(size, life)) * 0.2;
    } else {
        // Update velocity with forces
        vec3 acceleration = gravity + wind;
        outVelocity = velocity * damping + acceleration * deltaTime;
        
        // Update position
        outPosition = position + outVelocity * deltaTime;
        outSize = size;
    }
    
    // Transform feedback doesn't use gl_Position, but we set it for completeness
    gl_Position = vec4(outPosition, 1.0);
}