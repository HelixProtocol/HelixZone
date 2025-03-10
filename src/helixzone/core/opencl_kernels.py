"""Optimized OpenCL kernels for AMD GPUs."""

# Bilateral filter kernel optimized for AMD GPUs
# Uses local memory for tile-based processing and vectorized operations
BILATERAL_FILTER_KERNEL = """
#define TILE_SIZE 16
#define RADIUS_MAX 8
#define LOCAL_SIZE (TILE_SIZE + 2 * RADIUS_MAX)

__kernel void bilateral_filter(
    __global const uchar* input,
    __global float* output,
    const int width,
    const int height,
    const int d,
    const float sigma_color,
    const float sigma_space
) {
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    
    if (gx >= width || gy >= height)
        return;
        
    __local uchar local_data[LOCAL_SIZE][LOCAL_SIZE];
    const int radius = d / 2;
    const int tile_x = get_group_id(0) * TILE_SIZE - radius;
    const int tile_y = get_group_id(1) * TILE_SIZE - radius;
    
    // Load data into local memory
    for (int y = ly; y < LOCAL_SIZE; y += TILE_SIZE) {
        for (int x = lx; x < LOCAL_SIZE; x += TILE_SIZE) {
            const int px = tile_x + x;
            const int py = tile_y + y;
            local_data[y][x] = (px >= 0 && px < width && py >= 0 && py < height) ?
                input[py * width + px] : 0;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Process pixels using local memory
    const int local_x = lx + radius;
    const int local_y = ly + radius;
    const float center = local_data[local_y][local_x];
    float4 sum = (float4)(0.0f);
    float weight_sum = 0.0f;
    
    // Vectorized processing using float4
    for (int dy = -radius; dy <= radius; dy += 2) {
        for (int dx = -radius; dx <= radius; dx += 4) {
            const float4 pixels = (float4)(
                local_data[local_y + dy][local_x + dx],
                local_data[local_y + dy][local_x + dx + 1],
                local_data[local_y + dy][local_x + dx + 2],
                local_data[local_y + dy][local_x + dx + 3]
            );
            
            const float4 diff = pixels - center;
            const float4 space_dist = convert_float4(
                (dx * dx + dy * dy,
                 (dx + 1) * (dx + 1) + dy * dy,
                 (dx + 2) * (dx + 2) + dy * dy,
                 (dx + 3) * (dx + 3) + dy * dy)
            );
            
            const float4 weights = exp(-space_dist / (2.0f * sigma_space * sigma_space)) *
                                 exp(-diff * diff / (2.0f * sigma_color * sigma_color));
            
            sum += pixels * weights;
            weight_sum += weights.x + weights.y + weights.z + weights.w;
        }
    }
    
    output[gy * width + gx] = (sum.x + sum.y + sum.z + sum.w) / weight_sum;
}
"""

# Non-maximum suppression kernel optimized for AMD GPUs
# Uses local memory and vectorized operations
NMS_KERNEL = """
#define TILE_SIZE 16
#define LOCAL_SIZE (TILE_SIZE + 2)

__kernel void non_maximum_suppression(
    __global const float* strength,
    __global const float* angle,
    __global uchar* output,
    const int width,
    const int height
) {
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    
    if (gx >= width || gy >= height)
        return;
        
    __local float local_strength[LOCAL_SIZE][LOCAL_SIZE];
    __local float local_angle[LOCAL_SIZE][LOCAL_SIZE];
    const int tile_x = get_group_id(0) * TILE_SIZE - 1;
    const int tile_y = get_group_id(1) * TILE_SIZE - 1;
    
    // Load data into local memory
    for (int y = ly; y < LOCAL_SIZE; y += TILE_SIZE) {
        for (int x = lx; x < LOCAL_SIZE; x += TILE_SIZE) {
            const int px = tile_x + x;
            const int py = tile_y + y;
            if (px >= 0 && px < width && py >= 0 && py < height) {
                local_strength[y][x] = strength[py * width + px];
                local_angle[y][x] = angle[py * width + px];
            } else {
                local_strength[y][x] = 0.0f;
                local_angle[y][x] = 0.0f;
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Process pixels using local memory
    if (gx < 1 || gx >= width - 1 || gy < 1 || gy >= height - 1)
        return;
        
    const int local_x = lx + 1;
    const int local_y = ly + 1;
    const float val = local_strength[local_y][local_x];
    const int angle_quantized = (int)round(local_angle[local_y][local_x] * 4.0f / M_PI_F) & 3;
    
    float n1, n2;
    switch (angle_quantized) {
        case 0:  // -45 degrees
            n1 = local_strength[local_y - 1][local_x - 1];
            n2 = local_strength[local_y + 1][local_x + 1];
            break;
        case 1:  // vertical
            n1 = local_strength[local_y - 1][local_x];
            n2 = local_strength[local_y + 1][local_x];
            break;
        case 2:  // 45 degrees
            n1 = local_strength[local_y - 1][local_x + 1];
            n2 = local_strength[local_y + 1][local_x - 1];
            break;
        default:  // horizontal
            n1 = local_strength[local_y][local_x - 1];
            n2 = local_strength[local_y][local_x + 1];
            break;
    }
    
    output[gy * width + gx] = (val >= n1 && val >= n2) ? (uchar)val : 0;
}
"""

# Edge detection kernel optimized for AMD GPUs
# Uses local memory and vectorized operations
EDGE_DETECTION_KERNEL = """
#define TILE_SIZE 16
#define LOCAL_SIZE (TILE_SIZE + 2)

__kernel void sobel_edge_detection(
    __global const uchar* input,
    __global float* gradient_x,
    __global float* gradient_y,
    __global float* gradient_magnitude,
    __global float* gradient_angle,
    const int width,
    const int height
) {
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    
    if (gx >= width || gy >= height)
        return;
        
    __local float local_data[LOCAL_SIZE][LOCAL_SIZE];
    const int tile_x = get_group_id(0) * TILE_SIZE - 1;
    const int tile_y = get_group_id(1) * TILE_SIZE - 1;
    
    // Load data into local memory
    for (int y = ly; y < LOCAL_SIZE; y += TILE_SIZE) {
        for (int x = lx; x < LOCAL_SIZE; x += TILE_SIZE) {
            const int px = tile_x + x;
            const int py = tile_y + y;
            local_data[y][x] = (px >= 0 && px < width && py >= 0 && py < height) ?
                convert_float(input[py * width + px]) : 0.0f;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Process pixels using local memory
    if (gx < 1 || gx >= width - 1 || gy < 1 || gy >= height - 1)
        return;
        
    const int local_x = lx + 1;
    const int local_y = ly + 1;
    
    // Sobel operators
    const float gx_val =
        -local_data[local_y - 1][local_x - 1] + local_data[local_y - 1][local_x + 1] +
        -2.0f * local_data[local_y][local_x - 1] + 2.0f * local_data[local_y][local_x + 1] +
        -local_data[local_y + 1][local_x - 1] + local_data[local_y + 1][local_x + 1];
        
    const float gy_val =
        -local_data[local_y - 1][local_x - 1] - 2.0f * local_data[local_y - 1][local_x] - local_data[local_y - 1][local_x + 1] +
        local_data[local_y + 1][local_x - 1] + 2.0f * local_data[local_y + 1][local_x] + local_data[local_y + 1][local_x + 1];
    
    const int idx = gy * width + gx;
    gradient_x[idx] = gx_val;
    gradient_y[idx] = gy_val;
    gradient_magnitude[idx] = sqrt(gx_val * gx_val + gy_val * gy_val);
    gradient_angle[idx] = atan2(gy_val, gx_val);
}
""" 