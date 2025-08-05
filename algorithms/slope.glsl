#version 430

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, r32f) uniform restrict readonly image2D in_data;
layout(binding = 1, r32f) uniform restrict writeonly image2D out_data;

// Ignore pixels that have this value.
uniform float no_data;
// How big each pixel is in real-world units.
uniform vec2 texel_size;

const float sobel_x[9] = float[](
-1.0, 0.0, 1.0,
-2.0, 0.0, 2.0,
-1.0, 0.0, 1.0
);

const float sobel_y[9] = float[](
-1.0, -2.0, -1.0,
0.0, 0.0, 0.0,
1.0, 2.0, 1.0
);

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

    float dzdx = 0.0;
    float dzdy = 0.0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int x = gid.x + dx;
            int y = gid.y + dy;
            float filter_val = imageLoad(in_data, ivec2(x, y)).r;

            if (filter_val == no_data) {
                filter_val = val;
            }

            int k = (dy + 1) * 3 + (dx + 1);
            dzdx += filter_val * sobel_x[k];
            dzdy += filter_val * sobel_y[k];
        }
    }

    vec2 dz = vec2(dzdx, dzdy) / (8.0 * texel_size);

//    dzdx /= (8 * texel_size_x);
//    dzdy /= (8 * texel_size_x);

    float slope_percent = length(dz) * 100.0;

    // Compute aspect: angle clockwise from north (Y+ axis)
//    float aspect = degrees(atan(dzdx, dzdy)); // Note: atan(x, y)
//    if (aspect < 0.0) {
//        aspect += 360.0;
//    }

    imageStore(out_data, gid, vec4(slope_percent));
}
