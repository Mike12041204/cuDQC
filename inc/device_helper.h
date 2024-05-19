#ifndef DCUQC_DEVICE_HELPER_H
#define DCUQC_DEVICE_HELPER_H

#include "./common.h"

__device__ void d_sort(Vertex* target, int size, int (*func)(Vertex&, Vertex&));
__device__ void d_sort_i(int* target, int size, int (*func)(int, int));
__device__ int d_sort_vert_Q(Vertex& v1, Vertex& v2);
__device__ int d_sort_vert_cv(Vertex& v1, Vertex& v2);
__device__ int d_sort_degs(int n1, int n2);
__device__ int d_bsearch_array(int* search_array, int array_size, int search_number);
__device__ bool d_cand_isvalid_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ bool d_vert_isextendable_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_get_mindeg(int number_of_members, GPU_Data& dd);

#endif // DCUQC_DEVICE_HELPER_H