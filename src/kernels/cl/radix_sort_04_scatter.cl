#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define NUM_BINS 16

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* pref_sum,
    __global       uint* output,
    unsigned int bit,
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    if (((input[i] >> bit) & 1u) == 0){
        output[pref_sum[i] - 1] = input[i];
    } else {
        output[pref_sum[n-1] + (i - pref_sum[i])] = input[i];
    }
}
