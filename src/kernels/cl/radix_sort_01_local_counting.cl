#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define NUM_BINS 16

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input_data,
    __global       uint* output_data,
    unsigned int n,
    unsigned int bit_offset,
    unsigned int bits)
{
    //больше уже не нужно
}
