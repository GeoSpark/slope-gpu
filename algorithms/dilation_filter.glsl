#version 430

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, r32f) uniform restrict readonly image2D in_data;
layout(binding = 1, r32f) uniform restrict writeonly image2D out_data;

// Ignore pixels that have this value.
uniform float no_data;

float handle_nodata(float value) {
    if (value == no_data) {
        return 0.0;
    } else {
        return step(0.5, value);
    }
}


void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(in_data);

    // Skip edge pixels
    if (gid.x <= 0 || gid.y <= 0 || gid.x >= image_size.x - 1 || gid.y >= image_size.y - 1) {
        imageStore(out_data, gid, vec4(no_data));
        return;
    }

    float val = imageLoad(in_data, gid).r;

    if (val == no_data) {
        imageStore(out_data, gid, vec4(no_data));
        return;
    }

    float sum = 0.0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int x = gid.x + dx;
            int y = gid.y + dy;
            sum += handle_nodata(imageLoad(in_data, ivec2(x, y)).r);
        }
    }

    float output_val = sum > 0.0 ? 1.0 : 0.0;

    imageStore(out_data, gid, vec4(output_val));
}
