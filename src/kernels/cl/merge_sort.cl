#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    // TODO
    if (i >= n) {
        return;
    }
    unsigned int curr = input_data[i];
    unsigned int start = i - i % sorted_k;
    unsigned int end = start + sorted_k;

    const unsigned int mid = (start + end) / 2;
    
    unsigned int l, r;
    bool flag;
    
    if (i < mid) {
        flag = true;
        l = mid;
        r = end;
    } else {
        flag = false;
        l = start;
        r = mid;
    }
    
    while (l < r) {
        const unsigned int m = (l + r) / 2;
        bool eq;
        if (flag){
            eq = (m >= n) || (curr <= input_data[m]);
        } else {
            eq = (m >= n) || ((curr + 1) <= input_data[m]);
        }
        
        if (eq) {
            r = m;
        } else {
            l = m + 1;
        }
    }

    unsigned int ans_index = i + l - mid;
    if (ans_index < n) {
        output_data[ans_index] = curr;
    }
}
