#version 430

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, r32f) uniform restrict readonly image2D in_data;
layout(binding = 1, r32f) uniform restrict writeonly image2D out_data;

// Ignore pixels that have this value.
uniform float no_data;

void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);

    float val = imageLoad(in_data, gid).r;
    if (val == no_data || val == 0.0) {
        imageStore(out_data, gid, vec4(no_data));
        return;
    }
    imageStore(out_data, gid, vec4(val));
}
