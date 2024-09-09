#ifndef DCUQC_HOST_FUNCTIONS_H
#define DCUQC_HOST_FUNCTIONS_H

#include "./common.hpp"

// --- PRIMARY FUNCITONS --- 
void h_calculate_minimum_degrees(CPU_Graph& hg, int*& minimum_degrees, double minimum_degree_ratio);
void h_search(CPU_Graph& hg, ofstream& temp_results, DS_Sizes& dss, int* minimum_out_degrees, 
               int* minimum_in_degrees, double minimum_out_degree_ratio, 
               double minimum_in_degree_ratio, int minimum_clique_size, string output);
void h_allocate_memory(CPU_Data& hd, GPU_Data& h_dd, CPU_Cliques& hc, CPU_Graph& hg, 
                        DS_Sizes& dss, int* minimum_out_degrees, int* minimum_in_degrees, 
                        double minimum_out_degree_ratio, double minimum_in_degree_ratio, 
                        int minimum_clique_size);
void h_initialize_tasks(CPU_Graph& hg, CPU_Data& hd, int* minimum_out_degrees, 
                      int* minimum_in_degrees, int minimum_clique_size);
void h_expand_level(CPU_Graph& hg, CPU_Data& hd, CPU_Cliques& hc, DS_Sizes& dss, 
                    int* minimum_out_degrees, int* minimum_in_degrees, 
                    double minimum_out_degree_ratio, double minimum_in_degree_ratio, 
                    int minimum_clique_size);
void h_move_to_gpu(CPU_Data& hd, GPU_Data& h_dd, DS_Sizes& dss, string output);
void h_dump_cliques(CPU_Cliques& hc, GPU_Data& h_dd, ofstream& temp_results, DS_Sizes& dss);
void h_flush_cliques(CPU_Cliques& hc, ofstream& temp_results);
void h_free_memory(CPU_Data& hd, GPU_Data& h_dd, CPU_Cliques& hc);

// --- SECONDARY EXPANSION FUNCTIONS ---
int h_lookahead_pruning(CPU_Graph& hg, CPU_Cliques& hc, CPU_Data& hd, Vertex* read_vertices, 
                        int tot_vert, int num_mem, int num_cand, uint64_t start, 
                        int* minimum_degrees);
int h_remove_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* read_vertices, int& tot_vert, 
                        int& num_cand, int& num_mem, uint64_t start, int* minimum_out_degrees, 
                        int* minimum_in_degrees, int minimum_clique_size);
int h_add_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, 
                     int& number_of_candidates, int& number_of_members, int& upper_bound, 
                     int& lower_bound, int& min_ext_deg, int* minimum_out_degrees, 
                     int* minimum_in_degrees, double minimum_out_degree_ratio, 
                     double minimum_in_degree_ratio, int minimum_clique_size);
void h_diameter_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int pvertexid, 
                        int& total_vertices, int& number_of_candidates, int number_of_members, 
                        int min_out_deg, int min_in_deg);
bool h_degree_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, 
                    int& number_of_candidates, int number_of_members, int& upper_bound, 
                    int& lower_bound, int& min_ext_deg, int* minimum_out_degrees, 
                    int* minimum_in_degrees, double minimum_out_degree_ratio, 
                    double minimum_in_degree_ratio, int minimum_clique_size);
bool h_calculate_LU_bounds(CPU_Data& hd, int& upper_bound, int& lower_bound, int& min_ext_deg, 
                           Vertex* vertices, int number_of_members, int number_of_candidates, 
                           int* minimum_degrees, double minimum_degree_ratio, 
                           int minimum_clique_size);
void h_check_for_clique(CPU_Cliques& hc, Vertex* vertices, int number_of_members, 
                        int* minimum_out_degrees, int* minimum_in_degrees);
int h_critical_vertex_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, 
                              int& number_of_candidates, int& number_of_members, int& upper_bound, 
                              int& lower_bound, int& min_ext_deg, int* minimum_degrees, 
                              double minimum_degree_ratio, int minimum_clique_size);
void h_write_to_tasks(CPU_Data& hd, Vertex* vertices, int total_vertices, Vertex* write_vertices, 
                      uint64_t* write_offsets, uint64_t* write_count);
void h_fill_from_buffer(CPU_Data& hd, int threshold);

// --- TERTIARY FUNCTIONS ---
inline int h_comp_vert_Q(const void* a, const void* b)
{
    // order is: member -> covered -> cands -> cover
    // keys are: indeg -> exdeg -> lvl2adj -> vertexid
    
    Vertex* v1;
    Vertex* v2;
    int v1_mem_deg;
    int v2_mem_deg;
    int v1_can_deg;
    int v2_can_deg;

    v1 = (Vertex*)a;
    v2 = (Vertex*)b;

    v1_mem_deg = min(v1->out_mem_deg, v1->in_mem_deg);
    v2_mem_deg = min(v2->out_mem_deg, v2->in_mem_deg);
    v1_can_deg = min(v1->out_can_deg, v1->in_can_deg);
    v2_can_deg = min(v2->out_can_deg, v2->in_can_deg);

    if (v1->label == 1 && v2->label != 1)
        return -1;
    else if (v1->label != 1 && v2->label == 1)
        return 1;
    else if (v1->label == 2 && v2->label != 2)
        return -1;
    else if (v1->label != 2 && v2->label == 2)
        return 1;
    else if (v1->label == 0 && v2->label != 0)
        return -1;
    else if (v1->label != 0 && v2->label == 0)
        return 1;
    else if (v1->label == 3 && v2->label != 3)
        return -1;
    else if (v1->label != 3 && v2->label == 3)
        return 1;
    else if (v1_mem_deg > v2_mem_deg)
        return -1;
    else if (v1_mem_deg < v2_mem_deg)
        return 1;
    else if (v1_can_deg > v2_can_deg)
        return -1;
    else if (v1_can_deg < v2_can_deg)
        return 1;
    else if (v1->lvl2adj > v2->lvl2adj)
        return -1;
    else if (v1->lvl2adj < v2->lvl2adj)
        return 1;
    else if (v1->vertexid > v2->vertexid)
        return -1;
    else if (v1->vertexid < v2->vertexid)
        return 1;
    else
        return 0;
}
inline int h_comp_vert_cv(const void* a, const void* b)
{
    // put crit adj vertices before candidates

    Vertex* v1;
    Vertex* v2;

    v1 = (Vertex*)a;
    v2 = (Vertex*)b;

    if (v1->label == 4 && v2->label != 4)
        return -1;
    else if (v1->label != 4 && v2->label == 4)
        return 1;
    else
        return 0;
}
inline int h_comp_int_desc(const void* a, const void* b) 
{
    int n1;
    int n2;

    n1 = *(int*)a;
    n2 = *(int*)b;

    if (n1 > n2)
        return -1;
    else if (n1 < n2)
        return 1;
    else
        return 0;
}
inline int h_get_mindeg(int clique_size, int* minimum_degrees, int minimum_clique_size) {
    if (clique_size < minimum_clique_size)
        return minimum_degrees[minimum_clique_size];
    else
        return minimum_degrees[clique_size];
}
// TODO - update for bounds
inline bool h_cand_isvalid(Vertex vertex, int clique_size, int upper_bound, int lower_bound, 
                           int min_ext_deg, int* minimum_out_degrees, int* minimum_in_degrees, 
                           int minimum_clique_size) 
{
    if (vertex.out_mem_deg + vertex.out_can_deg < h_get_mindeg(clique_size + vertex.out_can_deg + 
        1, minimum_out_degrees, minimum_clique_size))
        return false;
    else if (vertex.in_mem_deg + vertex.in_can_deg < h_get_mindeg(clique_size + vertex.in_can_deg + 
        1, minimum_in_degrees, minimum_clique_size))
        return false;
    // else if (vertex.indeg + vertex.exdeg < min_ext_deg)
    //     return false;
    // else if (vertex.indeg + upper_bound - 1 < minimum_degrees[clique_size + upper_bound])
    //     return false;
    // else if (vertex.indeg + vertex.exdeg < h_get_mindeg(clique_size + lower_bound, minimum_degrees, minimum_clique_size))
    //     return false;
    else
        return true;
}
// TODO - update for bounds
inline bool h_vert_isextendable(Vertex vertex, int clique_size, int upper_bound, int lower_bound, 
                                int min_ext_deg, int* minimum_out_degrees, int* minimum_in_degrees, 
                                int minimum_clique_size)
{
    if (vertex.out_mem_deg + vertex.out_can_deg < h_get_mindeg(clique_size + vertex.out_can_deg, 
        minimum_out_degrees, minimum_clique_size))
        return false;
    else if (vertex.in_mem_deg + vertex.in_can_deg < h_get_mindeg(clique_size + vertex.in_can_deg, 
        minimum_in_degrees, minimum_clique_size))
        return false;
    // else if (vertex.indeg + vertex.exdeg < min_ext_deg)
    //     return false;
    // else if (vertex.indeg + upper_bound < minimum_degrees[clique_size + upper_bound])
    //     return false;
    // else if (vertex.indeg + vertex.exdeg < h_get_mindeg(clique_size + lower_bound, minimum_degrees, minimum_clique_size))
    //     return false;
    else
        return true;
}

#endif // DCUQC_HOST_FUNCTIONS_H