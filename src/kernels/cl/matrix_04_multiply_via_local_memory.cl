#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    // TODO
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int x_local = get_local_id(0);
    const unsigned int y_local = get_local_id(1);
    const unsigned int tile_size = 16;
    __local float tile_a[tile_size][tile_size + 1];
    __local float tile_b[tile_size][tile_size + 1];
    c[x + y * w] = 0.0f;

    for (unsigned int i = 0; i < (k - 1) / tile_size + 1; ++i){
        if (i * tile_size + x_local < k && y < h){
            tile_a[y_local][x_local] = a[y * k + i * tile_size + x_local];
        } else {
            tile_a[y_local][x_local] = 0.0f;
        }
        
        if (x < w && i * tile_size + y_local < k) {
            tile_b[y_local][x_local] = b[x + w * (i * tile_size + y_local)];
        } else {
            tile_b[y_local][x_local] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int j = 0; j < tile_size; ++j){
            c[x + y * w] += tile_a[y_local][j] * tile_b[j][x_local];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
