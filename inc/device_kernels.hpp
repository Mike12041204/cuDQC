#ifndef DCUQC_DEVICE_KERNELS_H
#define DCUQC_DEVICE_KERNELS_H

#include "./common.hpp"

// --- PRIMARY KERNELS ---
__global__ void d_expand_level(GPU_Data* dd);
__global__ void d_transfer_buffers(GPU_Data* dd, uint64_t* tasks_count, uint64_t* buffer_count, 
                                 uint64_t* cliques_count);
__global__ void d_fill_from_buffer(GPU_Data* dd, uint64_t* buffer_count);

// --- SECONDARY EXPANSION KERNELS ---
__device__ int d_lookahead_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_remove_one_vertex(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_add_one_vertex(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_critical_vertex_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_check_for_clique(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_write_to_tasks(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_diameter_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld, int pvertexid, 
                                   int min_out_deg, int min_in_deg);
__device__ void d_diameter_pruning_cv(GPU_Data* dd, Warp_Data& wd, Local_Data& ld, 
                                      int number_of_crit_adj);
__device__ void d_calculate_LU_bounds(GPU_Data* dd, Warp_Data& wd, Local_Data& ld, 
                                      int number_of_candidates);
__device__ bool d_degree_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);

// --- TERTIARY KERNELS ---
__device__ void d_oe_sort_vert(Vertex* target, int size, int (*func)(Vertex&, Vertex&));
__device__ void d_oe_sort_int(int* target, int size, int (*func)(int, int));
__device__ int d_b_search_int(int* search_array, int array_size, int search_number);
__device__ __forceinline__ int d_comp_vert_Q(Vertex& v1, Vertex& v2)
{
    // order is: member -> covered -> cands -> cover 
    // keys are: indeg -> exdeg -> lvl2adj -> vertexid

    int v1_mem_deg;
    int v2_mem_deg;
    int v1_can_deg;
    int v2_can_deg;

    v1_mem_deg = min(v1.out_mem_deg, v1.in_mem_deg);
    v2_mem_deg = min(v2.out_mem_deg, v2.in_mem_deg);
    v1_can_deg = min(v1.out_can_deg, v1.in_can_deg);
    v2_can_deg = min(v2.out_can_deg, v2.in_can_deg);

    if (v1.label == 1 && v2.label != 1)
        return -1;
    else if (v1.label != 1 && v2.label == 1)
        return 1;
    else if (v1.label == 2 && v2.label != 2)
        return -1;
    else if (v1.label != 2 && v2.label == 2)
        return 1;
    else if (v1.label == 0 && v2.label != 0)
        return -1;
    else if (v1.label != 0 && v2.label == 0)
        return 1;
    else if (v1.label == 3 && v2.label != 3)
        return -1;
    else if (v1.label != 3 && v2.label == 3)
        return 1;
    else if (v1_mem_deg > v2_mem_deg)
        return -1;
    else if (v1_mem_deg < v2_mem_deg)
        return 1;
    else if (v1_can_deg > v2_can_deg)
        return -1;
    else if (v1_can_deg < v2_can_deg)
        return 1;
    else if (v1.lvl2adj > v2.lvl2adj)
        return -1;
    else if (v1.lvl2adj < v2.lvl2adj)
        return 1;
    else if (v1.vertexid > v2.vertexid)
        return -1;
    else if (v1.vertexid < v2.vertexid)
        return 1;
    else
        return 0;
}
__device__ __forceinline__ int d_comp_vert_cv(Vertex& v1, Vertex& v2)
{
    // put crit adj vertices before candidates

    if (v1.label == 4 && v2.label != 4)
        return -1;
    else if (v1.label != 4 && v2.label == 4)
        return 1;
    else
        return 0;
}
__device__ __forceinline__ int d_comp_int_desc(int n1, int n2)
{
    // descending order

    if (n1 > n2)
        return -1;
    else if (n1 < n2)
        return 1;
    else
        return 0;
}
__device__ __forceinline__ int d_get_mindeg(int number_of_members, int* minimum_degrees, 
                                            int minimum_clique_size)
{
    if (number_of_members < minimum_clique_size)
        return minimum_degrees[minimum_clique_size];
    else
        return minimum_degrees[number_of_members];
}
// TODO - implement bounds
__device__ __forceinline__ bool d_cand_isvalid(Vertex& vertex, GPU_Data* dd, Warp_Data& wd, 
                                               Local_Data& ld)
{
    if (vertex.out_mem_deg + vertex.out_can_deg < d_get_mindeg(wd.number_of_members[WIB_IDX] + 
                                                               vertex.out_can_deg + 1, 
                                                               dd->minimum_out_degrees, 
                                                               *dd->minimum_clique_size))
        return false;
    else if (vertex.in_mem_deg + vertex.in_can_deg < d_get_mindeg(wd.number_of_members[WIB_IDX] + 
                                                                  vertex.in_can_deg + 1, 
                                                                  dd->minimum_in_degrees, 
                                                                  *dd->minimum_clique_size))
        return false;
    // else if (vertex.indeg + vertex.exdeg < wd.min_ext_deg[WIB_IDX])
    //     return false;
    // else if (vertex.indeg + wd.upper_bound[WIB_IDX] - 1 < dd->minimum_degrees[wd.number_of_members[WIB_IDX] + wd.upper_bound[WIB_IDX]])
    //     return false;
    // else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX], dd))
    //     return false;
    else
        return true;
}
// TODO - implement bounds
__device__ __forceinline__ bool d_vert_isextendable(Vertex& vertex, GPU_Data* dd, Warp_Data& wd, 
                                                    Local_Data& ld)
{
    if (vertex.out_mem_deg + vertex.out_can_deg < d_get_mindeg(wd.number_of_members[WIB_IDX] + 
                                                               vertex.out_can_deg, 
                                                               dd->minimum_out_degrees, 
                                                               *dd->minimum_clique_size))
        return false;
    else if (vertex.in_mem_deg + vertex.in_can_deg < d_get_mindeg(wd.number_of_members[WIB_IDX] + 
                                                                  vertex.in_can_deg, 
                                                                  dd->minimum_in_degrees, 
                                                                  *dd->minimum_clique_size))
        return false;
    // else if (vertex.indeg + vertex.exdeg < wd.min_ext_deg[WIB_IDX])
    //     return false;
    // else if (vertex.indeg + wd.upper_bound[WIB_IDX] < dd->minimum_degrees[wd.number_of_members[WIB_IDX] + wd.upper_bound[WIB_IDX]])
    //     return false;
    // else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX], dd))
    //     return false;
    else
        return true;
}

// --- DEBUG KERNELS ---
// __device__ void d_print_vertices(Vertex* vertices, int size);

#endif // DCUQC_DEVICE_KERNELS_H