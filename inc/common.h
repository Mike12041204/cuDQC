#ifndef DCUQC_COMMON_H
#define DCUQC_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <time.h>
#include <chrono>
#include <sys/timeb.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <sm_30_intrinsics.h>
#include <device_atomic_functions.h>
#include <mpi.h>
using namespace std;

// GPU KERNEL LAUNCH
#define BLOCK_SIZE 1024
#define NUM_OF_BLOCKS 216
#define WARP_SIZE 32

// GPU INFORMATION
#define IDX ((blockIdx.x * blockDim.x) + threadIdx.x)
#define WARP_IDX (IDX / WARP_SIZE)
#define LANE_IDX (IDX % WARP_SIZE)
#define WIB_IDX (threadIdx.x / WARP_SIZE)
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define NUMBER_OF_WARPS (NUM_OF_BLOCKS * WARPS_PER_BLOCK)
#define NUMBER_OF_THREADS (NUM_OF_BLOCKS * BLOCK_SIZE)

// DATA STRUCTURE SIZE
#define TASKS_SIZE 100000000
#define TASKS_PER_WARP 10
#define BUFFER_SIZE 100000000
#define BUFFER_OFFSET_SIZE 1000000
#define CLIQUES_SIZE 1000000
#define CLIQUES_OFFSET_SIZE 10000
#define CLIQUES_PERCENT 50
// per warp
#define WCLIQUES_SIZE 10000
#define WCLIQUES_OFFSET_SIZE 1000
#define WTASKS_SIZE 100000L
#define WTASKS_OFFSET_SIZE 10000
// global memory vertices, should be a multiple of 32 as to not waste space
#define WVERTICES_SIZE 32000
// shared memory vertices
#define VERTICES_SIZE 70

#define EXPAND_THRESHOLD (TASKS_PER_WARP * NUMBER_OF_WARPS)
#define CLIQUES_DUMP (CLIQUES_SIZE * (CLIQUES_PERCENT / 100.0))
 
// PROGRAM RUN SETTINGS
// cpu settings
#define CPU_LEVELS 1
#define CPU_EXPAND_THRESHOLD 1
// whether the program will run entirely on the CPU or not, 0-CPU/GPU 1-CPU only
#define CPU_MODE 0

// debug toggle 0-normal/1-debug
#define DEBUG_TOGGLE 1

// MPI SETTINGS
#define NUMBER_OF_PROCESSESS 4
#define MAX_MESSAGE 1000000000

// VERTEX DATA
struct Vertex
{
    int vertexid;
    // labels: 0 -> candidate, 1 -> member, 2 -> covered vertex, 3 -> cover vertex, 4 -> critical adjacent vertex
    int label;
    int indeg;
    int exdeg;
    int lvl2adj;
};

// CPU GRAPH / CONSTRUCTOR
class CPU_Graph
{
    public:

    int number_of_vertices;
    int number_of_edges;
    uint64_t number_of_lvl2adj;

    // one dimentional arrays of 1hop and 2hop neighbors and the offsets for each vertex
    int* onehop_neighbors;
    uint64_t* onehop_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;

    CPU_Graph(ifstream& graph_stream)
    {
        graph_stream >> number_of_vertices;
        graph_stream >> number_of_edges;
        graph_stream >> number_of_lvl2adj;

        onehop_neighbors = new int[number_of_edges];
        onehop_offsets = new uint64_t[number_of_vertices + 1];
        twohop_neighbors = new int[number_of_lvl2adj];
        twohop_offsets = new uint64_t[number_of_vertices + 1];

        for (int i = 0; i < number_of_edges; i++) {
            graph_stream >> onehop_neighbors[i];
        }

        for (int i = 0; i < number_of_vertices + 1; i++) {
            graph_stream >> onehop_offsets[i];
        }

        for (int i = 0; i < number_of_lvl2adj; i++) {
            graph_stream >> twohop_neighbors[i];
        }

        for (int i = 0; i < number_of_vertices + 1; i++) {
            graph_stream >> twohop_offsets[i];
        }
    }

    ~CPU_Graph() 
    {
        delete onehop_neighbors;
        delete onehop_offsets;
        delete twohop_neighbors;
        delete twohop_offsets;
    }
};

// CPU DATA
struct CPU_Data
{
    uint64_t* tasks1_count;
    uint64_t* tasks1_offset;
    Vertex* tasks1_vertices;

    uint64_t* tasks2_count;
    uint64_t* tasks2_offset;
    Vertex* tasks2_vertices;

    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    Vertex* buffer_vertices;

    uint64_t* current_level;
    bool* maximal_expansion;
    bool* dumping_cliques;

    int* vertex_order_map;
    int* remaining_candidates;
    int* removed_candidates;
    int* remaining_count;
    int* removed_count;
    int* candidate_indegs;
};

// CPU CLIQUES
struct CPU_Cliques
{
    uint64_t* cliques_count;
    uint64_t* cliques_offset;
    int* cliques_vertex;
};

// DEVICE DATA
struct GPU_Data
{
    // GPU DATA
    uint64_t* current_level;

    uint64_t* tasks1_count;
    uint64_t* tasks1_offset;
    Vertex* tasks1_vertices;

    uint64_t* tasks2_count;
    uint64_t* tasks2_offset;
    Vertex* tasks2_vertices;

    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    Vertex* buffer_vertices;

    uint64_t* wtasks_count;
    uint64_t* wtasks_offset;
    Vertex* wtasks_vertices;

    Vertex* global_vertices;

    int* removed_candidates;
    int* lane_removed_candidates;

    Vertex* remaining_candidates;
    int* lane_remaining_candidates;

    int* candidate_indegs;
    int* lane_candidate_indegs;

    int* adjacencies;

    int* total_tasks;

    double* minimum_degree_ratio;
    int* minimum_degrees;
    int* minimum_clique_size;
    int* scheduling_toggle;

    uint64_t* buffer_offset_start;
    uint64_t* buffer_start;
    uint64_t* cliques_offset_start;
    uint64_t* cliques_start;

    // GPU GRAPH
    int* number_of_vertices;
    int* number_of_edges;

    int* onehop_neighbors;
    uint64_t* onehop_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;

    // GPU CLIQUES
    uint64_t* cliques_count;
    uint64_t* cliques_offset;
    int* cliques_vertex;

    uint64_t* wcliques_count;
    uint64_t* wcliques_offset;
    int* wcliques_vertex;

    int* total_cliques;

    // moved from local
    Vertex* read_vertices;
    uint64_t* read_offsets;
    uint64_t* read_count;

    uint64_t* write_count;
    uint64_t* write_offsets;
    Vertex* write_vertices;

    // task scheduling
    int* current_task;
};

// WARP DATA
struct Warp_Data
{
    uint64_t start[WARPS_PER_BLOCK];
    uint64_t end[WARPS_PER_BLOCK];
    int tot_vert[WARPS_PER_BLOCK];
    int num_mem[WARPS_PER_BLOCK];
    int num_cand[WARPS_PER_BLOCK];
    int expansions[WARPS_PER_BLOCK];

    int number_of_members[WARPS_PER_BLOCK];
    int number_of_candidates[WARPS_PER_BLOCK];
    int total_vertices[WARPS_PER_BLOCK];

    Vertex shared_vertices[VERTICES_SIZE * WARPS_PER_BLOCK];

    int removed_count[WARPS_PER_BLOCK];
    int remaining_count[WARPS_PER_BLOCK];
    int num_val_cands[WARPS_PER_BLOCK];
    int rw_counter[WARPS_PER_BLOCK];

    int min_ext_deg[WARPS_PER_BLOCK];
    int lower_bound[WARPS_PER_BLOCK];
    int upper_bound[WARPS_PER_BLOCK];

    int tightened_upper_bound[WARPS_PER_BLOCK];
    int min_clq_indeg[WARPS_PER_BLOCK];
    int min_indeg_exdeg[WARPS_PER_BLOCK];
    int min_clq_totaldeg[WARPS_PER_BLOCK];
    int sum_clq_indeg[WARPS_PER_BLOCK];
    int sum_candidate_indeg[WARPS_PER_BLOCK];

    bool invalid_bounds[WARPS_PER_BLOCK];
    bool success[WARPS_PER_BLOCK];

    int number_of_crit_adj[WARPS_PER_BLOCK];

    // for dynamic intersection
    int count[WARPS_PER_BLOCK];
};

// LOCAL DATA
struct Local_Data
{
    Vertex* vertices;
};


#endif // DCUQC_COMMON_H