#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(__global const unsigned int* csr_row, __global const unsigned int* col, __global const unsigned int* values, __global const unsigned int* vector, __global unsigned int* output, unsigned int nrows) // TODO input/output buffers
{
    // TODO
    const unsigned int index = get_global_id(0);
    if (index < nrows){
        unsigned int start = csr_row[index];
        unsigned int end = csr_row[index + 1];
        unsigned int temp = 0;
        for (int i = start; i < end; ++i){
            temp += values[i] * vector[col[i]];
        }
        output[index] = temp;
    }
}
