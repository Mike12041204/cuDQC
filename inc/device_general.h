#ifndef DCUQC_DEVICE_GENERAL_H
#define DCUQC_DEVICE_GENERAL_H

#include "./common.h"

__global__ void d_expand_level(GPU_Data dd);
__global__ void transfer_buffers(GPU_Data dd);
__global__ void fill_from_buffer(GPU_Data dd);

__device__ int d_lookahead_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_remove_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_add_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_critical_vertex_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_check_for_clique(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_write_to_tasks(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_diameter_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int pvertexid);
__device__ void d_diameter_pruning_cv(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int number_of_crit_adj);
__device__ void d_calculate_LU_bounds(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int number_of_candidates);

__device__ bool d_degree_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_sort(Vertex* target, int size, int (*func)(Vertex&, Vertex&));
__device__ void d_sort_i(int* target, int size, int (*func)(int, int));
__device__ int d_sort_vert_Q(Vertex& v1, Vertex& v2);
__device__ int d_sort_vert_cv(Vertex& v1, Vertex& v2);
__device__ int d_sort_degs(int n1, int n2);
__device__ int d_bsearch_array(int* search_array, int array_size, int search_number);
__device__ bool d_cand_isvalid_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ bool d_vert_isextendable_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_get_mindeg(int number_of_members, GPU_Data& dd);

__device__ void d_print_vertices(Vertex* vertices, int size);

#endif // DCUQC_DEVICE_GENERAL_H