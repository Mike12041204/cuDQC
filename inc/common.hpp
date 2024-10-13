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
#include <cstring>
#include <time.h>
#include <chrono>
#include <sys/timeb.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <sm_30_intrinsics.h>
#include <device_atomic_functions.h>
//#include <mpi.h>
#include <omp.h>
#include <cassert>
#include <algorithm> 
#include <map>
//#include <pthread.h>
using namespace std;

// CPU DISTRIBUTED / PARALLEL SETTINGS
#define NUMBER_OF_PROCESSESS 1
#define NUMBER_OF_HTHREADS 16

// GPU KERNEL LAUNCH
#define BLOCK_SIZE 1024
#define NUMBER_OF_BLOCKS 216
#define WARP_SIZE 32

// GPU INFORMATION
#define IDX ((blockIdx.x * blockDim.x) + threadIdx.x)
#define WARP_IDX (IDX / WARP_SIZE)
#define LANE_IDX (IDX % WARP_SIZE)
#define TIB_IDX threadIdx.x
#define WIB_IDX (TIB_IDX / WARP_SIZE)
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define NUMBER_OF_WARPS (NUMBER_OF_BLOCKS * WARPS_PER_BLOCK)
#define NUMBER_OF_DTHREADS (NUMBER_OF_BLOCKS * BLOCK_SIZE)

// PROGRAM RUN SETTINGS
// shared memory vertices
#define VERTICES_SIZE 50
// cpu expansion settings
#define CPU_LEVELS 1
#define CPU_EXPAND_THRESHOLD 1
// mpi settings
#define MAX_MESSAGE 1000000000
// must be atleast be 1
#define HELP_MULTIPLIER 1
#define HELP_PERCENT 50
#define HELP_THRESHOLD (NUMBER_OF_WARPS * HELP_MULTIPLIER)

// VERTEX DATA
struct Vertex
{
    uint32_t vertexid;
    int8_t label;               // labels: 0 -> candidate, 1 -> member, 2 -> covered vertex, 3 -> cover vertex, 4 -> critical adjacent vertex, -1 -> pruned vertex
    uint32_t out_mem_deg;       // outgoing edge count to vertices in member (X) set
    uint32_t out_can_deg;       // outgoing edge count to vertices in candidate (Ext(X)) set
    uint32_t in_mem_deg;        // incoming edge count to vertices in member (X) set
    uint32_t in_can_deg;        // incoming edge count to vertices in candidate (Ext(X)) set
    uint32_t lvl2adj;           // count of vertices within twohops as defined by Guo paper
};

// CPU GRAPH / CONSTRUCTOR
class CPU_Graph
{
    public:

    int number_of_vertices;
    int number_of_edges;
    uint64_t number_of_lvl2adj;
    // one dimentional arrays of 1hop and 2hop neighbors and the offsets for each vertex
    int* out_neighbors;
    uint64_t* out_offsets;
    int* in_neighbors;
    uint64_t* in_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;

    CPU_Graph(ifstream& graph_stream);
    ~CPU_Graph();
    void GenLevel2NBs();
};

// CPU DATA
struct CPU_Data
{
    // data structures
    uint64_t* tasks1_count;
    uint64_t* tasks1_offset;
    Vertex* tasks1_vertices;
    uint64_t* tasks2_count;
    uint64_t* tasks2_offset;
    Vertex* tasks2_vertices;
    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    Vertex* buffer_vertices;
    // expansion helpers
    uint64_t* current_level;
    bool* maximal_expansion;
    // pruning helpers
    // TODO - see if some of these can be combined
    int* vertex_order_map;
    int* remaining_candidates;
    int* removed_candidates;
    int* remaining_count;
    int* removed_count;
    int* candidate_in_mem_degs;
    int* candidate_out_mem_degs;
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
    // data structures
    uint64_t* tasks_count;
    uint64_t* tasks_offset;
    Vertex* tasks_vertices;
    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    Vertex* buffer_vertices;
    // vertices
    Vertex* global_vertices;
    // warp data structures
    uint64_t* wtasks_count;
    uint64_t* wtasks_offset;
    Vertex* wtasks_vertices;
    // expansion helpers
    uint64_t* current_level;
    int* current_task;
    // count
    int* total_tasks;
    // pruning helpers
    // TODO - see if some of these can be combined
    int* removed_candidates;
    int* lane_removed_candidates;
    int* remaining_candidates;
    Vertex* temp_vertex_array;
    int* lane_remaining_candidates;
    int* candidate_in_mem_degs;
    int* lane_candidate_in_mem_degs;
    int* candidate_out_mem_degs;
    int* lane_candidate_out_mem_degs;
    int* adjacencies;
    int* vertex_order_map;
    // run parameters
    double* minimum_out_degree_ratio;
    int* minimum_out_degrees;
    double* minimum_in_degree_ratio;
    int* minimum_in_degrees;
    int* minimum_clique_size;
    // transfer helpers
    // TODO - remove
    uint64_t* buffer_offset_start;
    uint64_t* buffer_start;
    uint64_t* cliques_offset_start;
    uint64_t* cliques_start;
    // GPU GRAPH
    int* number_of_vertices;
    int* number_of_edges;
    int* out_neighbors;
    uint64_t* out_offsets;
    int* in_neighbors;
    uint64_t* in_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;
    // GPU CLIQUES
    // data structures
    uint64_t* cliques_count;
    uint64_t* cliques_offset;
    int* cliques_vertex;
    // warp data structures
    uint64_t* wcliques_count;
    uint64_t* wcliques_offset;
    int* wcliques_vertex;
    // count
    int* total_cliques;
    // DATA STRUCTURE SIZE
    uint64_t* TASKS_SIZE;
    uint64_t* TASKS_PER_WARP;
    uint64_t* BUFFER_SIZE;
    uint64_t* BUFFER_OFFSET_SIZE;
    uint64_t* CLIQUES_SIZE;
    uint64_t* CLIQUES_OFFSET_SIZE;
    uint64_t* CLIQUES_PERCENT;
    uint64_t* WCLIQUES_SIZE;
    uint64_t* WCLIQUES_OFFSET_SIZE;
    uint64_t* WTASKS_SIZE; //
    uint64_t* WTASKS_OFFSET_SIZE;
    uint64_t* WVERTICES_SIZE;
    uint64_t* EXPAND_THRESHOLD;
    uint64_t* CLIQUES_DUMP;
};

// WARP DATA
struct Warp_Data
{
    // previous level
    uint64_t start[WARPS_PER_BLOCK];
    uint64_t end[WARPS_PER_BLOCK];
    int tot_vert[WARPS_PER_BLOCK];
    int num_mem[WARPS_PER_BLOCK];
    int num_cand[WARPS_PER_BLOCK];
    int expansions[WARPS_PER_BLOCK];
    // next level
    int number_of_members[WARPS_PER_BLOCK];
    int number_of_candidates[WARPS_PER_BLOCK];
    int total_vertices[WARPS_PER_BLOCK];
    Vertex shared_vertices[VERTICES_SIZE * WARPS_PER_BLOCK];
    // pruning helpers
    int removed_count[WARPS_PER_BLOCK];
    int remaining_count[WARPS_PER_BLOCK];
    int num_val_cands[WARPS_PER_BLOCK];
    int success[WARPS_PER_BLOCK];
    // bounds
    int min_ext_out_deg[WARPS_PER_BLOCK];
    int min_ext_in_deg[WARPS_PER_BLOCK];
    int lower_bound[WARPS_PER_BLOCK];
    int upper_bound[WARPS_PER_BLOCK];
    // bound helpers
    int nmin_clq_clqdeg_o[WARPS_PER_BLOCK];
    int nminclqdeg_candeg_o[WARPS_PER_BLOCK];
    int nmin_clq_totaldeg_o[WARPS_PER_BLOCK];
    int nclq_clqdeg_sum_o[WARPS_PER_BLOCK];
    int ncand_clqdeg_sum_o[WARPS_PER_BLOCK];
    int nmin_clq_clqdeg_i[WARPS_PER_BLOCK];
    int nminclqdeg_candeg_i[WARPS_PER_BLOCK];
    int nmin_clq_totaldeg_i[WARPS_PER_BLOCK];
    int nclq_clqdeg_sum_i[WARPS_PER_BLOCK];
    int ncand_clqdeg_sum_i[WARPS_PER_BLOCK];
};

// LOCAL DATA
struct Local_Data
{
    Vertex* vertices;
};

// DATA STRUCTURE SIZES
class DS_Sizes
{
    public:
    
    // DATA STRUCTURE SIZE
    uint64_t TASKS_SIZE;
    uint64_t TASKS_PER_WARP;
    uint64_t BUFFER_SIZE;
    uint64_t BUFFER_OFFSET_SIZE;
    uint64_t CLIQUES_SIZE;
    uint64_t CLIQUES_OFFSET_SIZE;
    uint64_t CLIQUES_PERCENT;
    // per warp
    uint64_t WCLIQUES_SIZE;
    uint64_t WCLIQUES_OFFSET_SIZE;
    uint64_t WTASKS_SIZE;
    uint64_t WTASKS_OFFSET_SIZE;
    // global memory vertices, should be a multiple of 32 as to not waste space
    uint64_t WVERTICES_SIZE;
    uint64_t EXPAND_THRESHOLD;
    uint64_t CLIQUES_DUMP;
    int DEBUG_TOGGLE;

    DS_Sizes(const string& filename);
};

// DEBUG - MAX TRACKER VARIABLES
extern uint64_t mts, mbs, mbo, mcs, mco, wts, wto, wcs, wco, mvs;
extern ofstream output_file;

// cuTS MPI VARIABLES
extern int wsize;
extern int grank;
extern char msg_buffer[NUMBER_OF_PROCESSESS][100];             // for every task there is a seperate message buffer and incoming/outgoing handle slot
// DEBUG - uncomment
//extern MPI_Request rq_send_msg[NUMBER_OF_PROCESSESS];          // array of handles for messages with all other thread, allows for asynchronous messaging, handles say whether message is complete
//extern MPI_Request rq_recv_msg[NUMBER_OF_PROCESSESS];
extern bool global_free_list[NUMBER_OF_PROCESSESS];

// DEBUG - rm
extern uint64_t db0, db1, db2, db3;

inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        cout << cudaGetErrorString(code) << endl;
        exit(-1);
    }
}

inline int comp_int(const void *e1, const void *e2)
{
	int n1, n2;
	n1 = *(int *) e1;
	n2 = *(int *) e2;

	if (n1>n2)
		return 1;
	else if (n1<n2)
		return -1;
	else
		return 0;
}

#endif // DCUQC_COMMON_H