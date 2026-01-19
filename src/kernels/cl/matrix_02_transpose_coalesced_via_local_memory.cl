#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    // TODO
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int x_local = get_local_id(0);
    const unsigned int y_local = get_local_id(1);

    __local float local_data[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    if (x < w && y < h){
        local_data[y_local][x_local] = matrix[x + y * w];
    } else {
        local_data[y_local][x_local] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const unsigned int x_new = y - y_local + x_local;
    const unsigned int y_new = x - x_local + y_local;

    if (x_new < h && y_new < w){
        transposed_matrix[y_new * h + x_new] = local_data[x_local][y_local];
    }
}
