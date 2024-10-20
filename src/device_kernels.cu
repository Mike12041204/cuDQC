#include "../inc/common.hpp"
#include "../inc/device_kernels.hpp"

// --- PRIMARY KERNELS ---

__global__ void d_expand_level(GPU_Data* dd)
{
    __shared__ Warp_Data wd;        // data is stored in data structures to reduce the number of variables that need to be passed to methods
    Local_Data ld;
    int num_mem;                    // helper variables, not passed through to any methods
    int index;

    // reset warp tasks and cliques counts
    if (LANE_IDX == 0) {
        dd->wtasks_count[WARP_IDX] = 0;
        dd->wcliques_count[WARP_IDX] = 0;
    }
    __syncwarp();

    // --- CURRENT LEVEL ---

    // initialize i for each warp
    int i = WARP_IDX;
    
    while (i < *dd->tasks_count) 
    {

        // INITIALIZE OLD VERTICES
        // get information on vertices being handled within tasks
        if (LANE_IDX == 0) {
            wd.start[WIB_IDX] = dd->tasks_offset[i];
            wd.end[WIB_IDX] = dd->tasks_offset[i + 1];
            wd.tot_vert[WIB_IDX] = wd.end[WIB_IDX] - wd.start[WIB_IDX];
        }
        __syncwarp();

        // each warp gets partial number of members
        num_mem = 0;
        for (uint64_t j = wd.start[WIB_IDX] + LANE_IDX; j < wd.end[WIB_IDX]; j += WARP_SIZE) {
            if (dd->tasks_vertices[j].label != 1) {
                break;
            }
            num_mem++;
        }
        // sum members across warp
        for (int j = 1; j < 32; j *= 2) {
            num_mem += __shfl_xor_sync(0xFFFFFFFF, num_mem, j);
        }

        if (LANE_IDX == 0) {
            wd.num_mem[WIB_IDX] = num_mem;
            wd.num_cand[WIB_IDX] = wd.tot_vert[WIB_IDX] - wd.num_mem[WIB_IDX];
            wd.expansions[WIB_IDX] = wd.num_cand[WIB_IDX];
        }

        // LOOKAHEAD PRUNING
        wd.success[WIB_IDX] = true;
        __syncwarp();

        // sets success to false if lookahead fails
        d_lookahead_pruning(dd, wd, ld);
        
        if (wd.success[WIB_IDX]) {
            // schedule warps next task
            if (LANE_IDX == 0) {
                i = atomicAdd(dd->current_task, 1);
            }
            i = __shfl_sync(0xFFFFFFFF, i, 0);
            continue;
        }

        // --- NEXT LEVEL ---

        for (int j = 0; j < wd.expansions[WIB_IDX]; j++)
        {
            
            // REMOVE ONE VERTEX
            if (j > 0) {
                if(LANE_IDX == 0){
                    wd.success[WIB_IDX] = true;
                }
                __syncwarp();

                // sets success to false is failed vertex found, sets to 2 if next vertex to be
                // added is a failed vertex
                d_remove_one_vertex(dd, wd, ld);

                if (!wd.success[WIB_IDX]) {
                    break;
                }
                if(wd.success[WIB_IDX] == 2){
                    continue;
                }
            }

            // INITIALIZE NEW VERTICES
            if (LANE_IDX == 0) {
                wd.number_of_members[WIB_IDX] = wd.num_mem[WIB_IDX];
                wd.number_of_candidates[WIB_IDX] = wd.num_cand[WIB_IDX];
                wd.total_vertices[WIB_IDX] = wd.tot_vert[WIB_IDX];
            }
            __syncwarp();

            // select whether to store vertices in global or shared memory based on size
            if (wd.total_vertices[WIB_IDX] <= VERTICES_SIZE) {
                ld.vertices = wd.shared_vertices + (VERTICES_SIZE * WIB_IDX);
            }
            else {
                ld.vertices = dd->global_vertices + (*dd->WVERTICES_SIZE * WARP_IDX);
            }

            // copy vertices
            for (index = LANE_IDX; index < wd.number_of_members[WIB_IDX]; index += WARP_SIZE) {
                ld.vertices[index] = dd->tasks_vertices[wd.start[WIB_IDX] + index];
            }
            for (; index < wd.total_vertices[WIB_IDX] - 1; index += WARP_SIZE) {
                ld.vertices[index + 1] = dd->tasks_vertices[wd.start[WIB_IDX] + index];
            }
            if (LANE_IDX == 0) {
                ld.vertices[wd.number_of_members[WIB_IDX]] = dd->tasks_vertices[wd.start[WIB_IDX] + 
                    wd.total_vertices[WIB_IDX] - 1];
            }
            __syncwarp();

            // ADD ONE VERTEX
            if(LANE_IDX == 0){
                wd.success[WIB_IDX] = true;
            }
            __syncwarp();
            
            // sets success to false if failed found
            d_add_one_vertex(dd, wd, ld);

            // if failed found check for clique and continue on to the next iteration
            if (!wd.success[WIB_IDX]) {
                d_check_for_clique(dd, wd, ld);
                
                continue;
            }

            // CRITICAL VERTEX PRUNING
            if(LANE_IDX == 0){
                wd.success[WIB_IDX] = 1;
            }
            __syncwarp();

            // sets success as 2 if critical fail, 0 if failed found or invalid bound, 1 otherwise
            d_critical_vertex_pruning(dd, wd, ld);

            // critical fail, cannot be clique continue onto next iteration
            if (wd.success[WIB_IDX] == 2) {
                continue;
            }

            // HANDLE CLIQUES
            d_check_for_clique(dd, wd, ld);

            // if vertex in x found as not extendable continue to next iteration
            if (wd.success[WIB_IDX] == 0) {
                continue;
            }

            // WRITE TASKS TO BUFFERS
            // sort vertices in Quick efficient enumeration order before writing
            d_oe_sort_vert(ld.vertices, wd.total_vertices[WIB_IDX], d_comp_vert_Q);

            if (wd.number_of_candidates[WIB_IDX] > 0) {
                d_write_to_tasks(dd, wd, ld);
            }
        }

        // schedule warps next task
        if (LANE_IDX == 0) {
            i = atomicAdd(dd->current_task, 1);
        }
        i = __shfl_sync(0xFFFFFFFF, i, 0);
    }

    // TODO - make each block just sum this themselves without atomic operation
    if (LANE_IDX == 0) {
        // sum to find tasks count
        atomicAdd(dd->total_tasks, dd->wtasks_count[WARP_IDX]);
        atomicAdd(dd->total_cliques, dd->wcliques_count[WARP_IDX]);
    }

    // TODO - this should be easy to remove and just make local in transfer_buffers
    if (IDX == 0) {
        *dd->buffer_offset_start = *dd->buffer_count + 1;
        *dd->buffer_start = dd->buffer_offset[*dd->buffer_count];
        *dd->cliques_offset_start = *dd->cliques_count + 1;
        *dd->cliques_start = dd->cliques_offset[*dd->cliques_count];
    }
}

__global__ void d_transfer_buffers(GPU_Data* dd, uint64_t* tasks_count, uint64_t* buffer_count, 
                                   uint64_t* cliques_count, uint64_t* cliques_size)
{
    __shared__ uint64_t tasks_write[WARPS_PER_BLOCK];
    __shared__ int tasks_offset_write[WARPS_PER_BLOCK];
    __shared__ uint64_t cliques_write[WARPS_PER_BLOCK];
    __shared__ int cliques_offset_write[WARPS_PER_BLOCK];
    __shared__ int twarp;
    __shared__ int toffsetwrite;
    __shared__ int twrite;
    __shared__ int tasks_end;

    // point of this is to find how many vertices will be transfered to tasks, it is easy to know how many tasks as it will just
    // be the expansion threshold, but to find how many vertices we must now the total size of all the tasks that will be copied.
    // each block does this but really could be done by one thread outside the GPU
    if (TIB_IDX == 0) {
        toffsetwrite = 0;
        twrite = 0;

        for (int i = 0; i < NUMBER_OF_WARPS; i++) {
            // if next warps count is more than expand threshold mark as such and break
            if (toffsetwrite + dd->wtasks_count[i] >= *dd->EXPAND_THRESHOLD) {
                twarp = i;
                break;
            }
            // else adds its size and count
            twrite += dd->wtasks_offset[(*dd->WTASKS_OFFSET_SIZE * i) + dd->wtasks_count[i]];
            toffsetwrite += dd->wtasks_count[i];
        }
        // final size is the size of all tasks up until last warp and the remaining tasks in the last warp until expand threshold is satisfied
        tasks_end = twrite + dd->wtasks_offset[(*dd->WTASKS_OFFSET_SIZE * twarp) + (*dd->EXPAND_THRESHOLD - toffsetwrite)];
    }
    __syncthreads();

    // warp level
    if (LANE_IDX == 0) {
        tasks_write[WIB_IDX] = 0;
        tasks_offset_write[WIB_IDX] = 1;
        cliques_write[WIB_IDX] = 0;
        cliques_offset_write[WIB_IDX] = 1;

        for (int i = 0; i < WARP_IDX; i++) {
            tasks_offset_write[WIB_IDX] += dd->wtasks_count[i];
            tasks_write[WIB_IDX] += dd->wtasks_offset[(*dd->WTASKS_OFFSET_SIZE * i) + dd->wtasks_count[i]];

            cliques_offset_write[WIB_IDX] += dd->wcliques_count[i];
            cliques_write[WIB_IDX] += dd->wcliques_offset[(*dd->WCLIQUES_OFFSET_SIZE * i) + dd->wcliques_count[i]];
        }
    }
    __syncwarp();
    
    // move to tasks and buffer
    for (uint64_t i = LANE_IDX + 1; i <= dd->wtasks_count[WARP_IDX]; i += WARP_SIZE) {
        if (tasks_offset_write[WIB_IDX] + i - 1 <= *dd->EXPAND_THRESHOLD) {
            // to tasks
            dd->tasks_offset[tasks_offset_write[WIB_IDX] + i - 1] = dd->wtasks_offset[(*dd->WTASKS_OFFSET_SIZE * WARP_IDX) + i] + tasks_write[WIB_IDX];
        }
        else {
            // to buffer
            dd->buffer_offset[tasks_offset_write[WIB_IDX] + i - 2 - *dd->EXPAND_THRESHOLD + *dd->buffer_offset_start] = dd->wtasks_offset[(*dd->WTASKS_OFFSET_SIZE * WARP_IDX) + i] +
                tasks_write[WIB_IDX] - tasks_end + *dd->buffer_start;
        }
    }

    for (uint64_t i = LANE_IDX; i < dd->wtasks_offset[(*dd->WTASKS_OFFSET_SIZE * WARP_IDX) + dd->wtasks_count[WARP_IDX]]; i += WARP_SIZE) {
        if (tasks_write[WIB_IDX] + i < tasks_end) {
            // to tasks
            dd->tasks_vertices[tasks_write[WIB_IDX] + i] = dd->wtasks_vertices[(*dd->WTASKS_SIZE * WARP_IDX) + i];
        }
        else {
            // to buffer
            dd->buffer_vertices[*dd->buffer_start + tasks_write[WIB_IDX] + i - tasks_end] = dd->wtasks_vertices[(*dd->WTASKS_SIZE * WARP_IDX) + i];
        }
    }
    // NOTE - this sync is important for some reason, larger graphs/et dont work without it
    __syncthreads();

    //move to cliques
    for (uint64_t i = LANE_IDX + 1; i <= dd->wcliques_count[WARP_IDX]; i += WARP_SIZE) {
        dd->cliques_offset[*dd->cliques_offset_start + cliques_offset_write[WIB_IDX] + i - 2] = dd->wcliques_offset[(*dd->WCLIQUES_OFFSET_SIZE * WARP_IDX) + i] + *dd->cliques_start + 
            cliques_write[WIB_IDX];
    }
    for (uint64_t i = LANE_IDX; i < dd->wcliques_offset[(*dd->WCLIQUES_OFFSET_SIZE * WARP_IDX) + dd->wcliques_count[WARP_IDX]]; i += WARP_SIZE) {
        dd->cliques_vertex[*dd->cliques_start + cliques_write[WIB_IDX] + i] = dd->wcliques_vertex[(*dd->WCLIQUES_SIZE * WARP_IDX) + i];
    }

    if (IDX == NUMBER_OF_DTHREADS - 1) {
        // handle tasks and buffer counts
        if (*dd->total_tasks <= *dd->EXPAND_THRESHOLD) {
            *dd->tasks_count = *dd->total_tasks;
        }
        else {
            *dd->tasks_count = *dd->EXPAND_THRESHOLD;
            *dd->buffer_count += *dd->total_tasks - *dd->EXPAND_THRESHOLD;
        }
        *dd->cliques_count += *dd->total_cliques;

        *dd->total_tasks = 0;
        *dd->total_cliques = 0;
        (*dd->current_level)++;

        *dd->current_task = NUMBER_OF_WARPS;
        *tasks_count = *dd->tasks_count;
        *buffer_count = *dd->buffer_count;
        *cliques_count = *dd->cliques_count;
        *cliques_size = dd->cliques_offset[*cliques_count];
    }
}

__global__ void d_fill_from_buffer(GPU_Data* dd, uint64_t* tasks_count, uint64_t* buffer_count)
{
    // get read and write locations
    uint64_t write_amount = (*dd->buffer_count >= *dd->EXPAND_THRESHOLD - *dd->tasks_count) ? *dd->EXPAND_THRESHOLD - *dd->tasks_count : *dd->buffer_count;
    uint64_t start_buffer = dd->buffer_offset[*dd->buffer_count - write_amount];
    uint64_t end_buffer = dd->buffer_offset[*dd->buffer_count];
    uint64_t size_buffer = end_buffer - start_buffer;
    uint64_t start_write = dd->tasks_offset[*dd->tasks_count];

    // handle offsets
    for (uint64_t i = IDX + 1; i <= write_amount; i += NUMBER_OF_DTHREADS) {
        dd->tasks_offset[*dd->tasks_count + i] = start_write + dd->buffer_offset[*dd->buffer_count - write_amount + i] - start_buffer;
    }

    // handle data
    for (uint64_t i = IDX; i < size_buffer; i += NUMBER_OF_DTHREADS) {
        dd->tasks_vertices[start_write + i] = dd->buffer_vertices[start_buffer + i];
    }

    if (IDX == 0) {
        *dd->tasks_count += write_amount;
        *dd->buffer_count -= write_amount;

        *buffer_count = *dd->buffer_count;
        *tasks_count = *dd->tasks_count;
    }
}

// --- SECONDARY EXPANSION KERNELS ---
// sets success to false if lookahead fails
__device__ void d_lookahead_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    int pvertexid;
    int phelper1;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    uint64_t start_write;
    int min_out_deg;
    int min_in_deg;
    uint64_t warp_write;

    warp_write = WARP_IDX * *dd->WVERTICES_SIZE;

    min_out_deg = d_get_mindeg(wd.tot_vert[WIB_IDX], dd->minimum_out_degrees, 
                               *dd->minimum_clique_size);
    min_in_deg = d_get_mindeg(wd.tot_vert[WIB_IDX], dd->minimum_in_degrees, 
                               *dd->minimum_clique_size);

    // check if members meet degree requirement, dont need to check 2hop adj as diameter pruning 
    // guarentees all members will be within 2hops of eveything
    for (int i = LANE_IDX; i < wd.num_mem[WIB_IDX] && wd.success[WIB_IDX]; i += WARP_SIZE) {
        if (dd->tasks_vertices[wd.start[WIB_IDX] + i].out_mem_deg + 
            dd->tasks_vertices[wd.start[WIB_IDX] + i].out_can_deg < min_out_deg || 
            dd->tasks_vertices[wd.start[WIB_IDX] + i].in_mem_deg + 
            dd->tasks_vertices[wd.start[WIB_IDX] + i].in_can_deg < min_in_deg) {

            wd.success[WIB_IDX] = false;
            break;
        }
    }
    __syncwarp();

    if (!wd.success[WIB_IDX]) {
        return;
    }

    // initialize vertex order map
    for(int i = LANE_IDX; i < wd.tot_vert[WIB_IDX]; i += WARP_SIZE){
        dd->vertex_order_map[warp_write + dd->tasks_vertices[wd.start[WIB_IDX] + i].vertexid] = i;
    }

    for (int i = wd.num_mem[WIB_IDX]; i < wd.tot_vert[WIB_IDX]; i++) {

        pvertexid = dd->tasks_vertices[wd.start[WIB_IDX] + i].vertexid;

        pneighbors_start = dd->twohop_offsets[pvertexid];
        pneighbors_end = dd->twohop_offsets[pvertexid + 1];

        for (int j = pneighbors_start + LANE_IDX; j < pneighbors_end; j += WARP_SIZE) {

            phelper1 = dd->vertex_order_map[warp_write + dd->twohop_neighbors[j]];

            if (phelper1 >= wd.num_mem[WIB_IDX]) {
                dd->tasks_vertices[wd.start[WIB_IDX] + phelper1].lvl2adj++;
            }
        }
        __syncwarp();
    }

    // reset vertex order map
    for(int i = LANE_IDX; i < wd.tot_vert[WIB_IDX]; i += WARP_SIZE){
        dd->vertex_order_map[warp_write + dd->tasks_vertices[wd.start[WIB_IDX] + i].vertexid] = -1;
    }
    __syncwarp();

    // compares all vertices to the lemmas from Quick
    for (int i = wd.num_mem[WIB_IDX] + LANE_IDX; i < wd.tot_vert[WIB_IDX] && wd.success[WIB_IDX]; 
         i += WARP_SIZE) {

        if (dd->tasks_vertices[wd.start[WIB_IDX] + i].lvl2adj < wd.num_cand[WIB_IDX] - 1 || 
            dd->tasks_vertices[wd.start[WIB_IDX] + i].out_mem_deg + 
            dd->tasks_vertices[wd.start[WIB_IDX] + i].out_can_deg < min_out_deg || 
            dd->tasks_vertices[wd.start[WIB_IDX] + i].in_mem_deg + 
            dd->tasks_vertices[wd.start[WIB_IDX] + i].in_can_deg < min_in_deg){

            wd.success[WIB_IDX] = false;
            break;
        }
    }
    __syncwarp();
    if (!wd.success[WIB_IDX]) {
        return;
    }

    // write to cliques
    start_write = (*dd->WCLIQUES_SIZE * WARP_IDX) + dd->wcliques_offset[(*dd->WCLIQUES_OFFSET_SIZE * 
        WARP_IDX) + dd->wcliques_count[WARP_IDX]];
    for (int i = LANE_IDX; i < wd.tot_vert[WIB_IDX]; i += WARP_SIZE) {
        dd->wcliques_vertex[start_write + i] = dd->tasks_vertices[wd.start[WIB_IDX] + i].vertexid;
    }
    if (LANE_IDX == 0) {
        (dd->wcliques_count[WARP_IDX])++;
        dd->wcliques_offset[(*dd->WCLIQUES_OFFSET_SIZE * WARP_IDX) + dd->wcliques_count[WARP_IDX]] = 
            start_write - (*dd->WCLIQUES_SIZE * WARP_IDX) + wd.tot_vert[WIB_IDX];
    }
    __syncwarp();
}

// sets success to false if failed found else it remains true
__device__ void d_remove_one_vertex(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    int pvertexid;
    int phelper1;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int min_out_deg;
    int min_in_deg;
    int warp_write;

    warp_write = WARP_IDX * *dd->WVERTICES_SIZE;

    min_out_deg = d_get_mindeg(wd.num_mem[WIB_IDX] + 1, dd->minimum_out_degrees, 
                               *dd->minimum_clique_size);
    min_in_deg = d_get_mindeg(wd.num_mem[WIB_IDX] + 1, dd->minimum_in_degrees, 
                               *dd->minimum_clique_size);

    // remove the last candidate in vertices
    if (LANE_IDX == 0) {
        wd.num_cand[WIB_IDX]--;
        wd.tot_vert[WIB_IDX]--;
    }
    __syncwarp();

    // initialize vertex order map
    for(int i = LANE_IDX; i < wd.tot_vert[WIB_IDX]; i += WARP_SIZE){
        dd->vertex_order_map[warp_write + dd->tasks_vertices[wd.start[WIB_IDX] + i].vertexid] = i;
    }
    __syncwarp();

    // update info of vertices connected to removed cand
    pvertexid = dd->tasks_vertices[wd.start[WIB_IDX] + wd.tot_vert[WIB_IDX]].vertexid;

    pneighbors_start = dd->out_offsets[pvertexid];
    pneighbors_end = dd->out_offsets[pvertexid + 1];

    for (uint64_t i = pneighbors_start + LANE_IDX; i < pneighbors_end && wd.success[WIB_IDX]; i += 
         WARP_SIZE) {

        phelper1 = dd->vertex_order_map[warp_write + dd->out_neighbors[i]];

        if (phelper1 > -1) {
            dd->tasks_vertices[wd.start[WIB_IDX] + phelper1].in_can_deg--;

            if (dd->tasks_vertices[wd.start[WIB_IDX] + phelper1].in_mem_deg + 
                dd->tasks_vertices[wd.start[WIB_IDX] + phelper1].in_can_deg < 
                min_in_deg) {
                
                if(phelper1 < wd.num_mem[WIB_IDX]){
                    wd.success[WIB_IDX] = false;
                    break;
                }
                else if(phelper1 == wd.tot_vert[WIB_IDX] - 1){
                    wd.success[WIB_IDX] = 2;
                }
            }
        }
    }
    __syncwarp();

    if (!wd.success[WIB_IDX]) {
        // reset vertex order map
        for(int i = LANE_IDX; i < wd.tot_vert[WIB_IDX]; i += WARP_SIZE){
            dd->vertex_order_map[warp_write + dd->tasks_vertices[wd.start[WIB_IDX] + i].vertexid] = -1;
        }

        return;
    }

    pneighbors_start = dd->in_offsets[pvertexid];
    pneighbors_end = dd->in_offsets[pvertexid + 1];

    for (uint64_t i = pneighbors_start + LANE_IDX; i < pneighbors_end && wd.success[WIB_IDX]; i += 
         WARP_SIZE) {

        phelper1 = dd->vertex_order_map[warp_write + dd->in_neighbors[i]];

        if (phelper1 > -1) {
            dd->tasks_vertices[wd.start[WIB_IDX] + phelper1].out_can_deg--;

            if (dd->tasks_vertices[wd.start[WIB_IDX] + phelper1].out_mem_deg + 
                dd->tasks_vertices[wd.start[WIB_IDX] + phelper1].out_can_deg < 
                min_out_deg) {
                
                if(phelper1 < wd.num_mem[WIB_IDX]){
                    wd.success[WIB_IDX] = false;
                    break;
                }
                else if(phelper1 == wd.tot_vert[WIB_IDX] - 1){
                    wd.success[WIB_IDX] = 2;
                }
            }
        }
    }

    // reset vertex order map
    for(int i = LANE_IDX; i < wd.tot_vert[WIB_IDX]; i += WARP_SIZE){
        dd->vertex_order_map[warp_write + dd->tasks_vertices[wd.start[WIB_IDX] + i].vertexid] = -1;
    }
    __syncwarp();
}

// sets success to false if failed found else leaves as true
__device__ void d_add_one_vertex(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    int pvertexid;
    int phelper1;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int min_out_deg;
    int min_in_deg;
    uint64_t warp_write;

    warp_write = WARP_IDX * *dd->WVERTICES_SIZE;

    // minimum degrees for candidates, plus two for vertex that will be added and candidates itself
    // will need to be added as well
    min_out_deg = d_get_mindeg(wd.number_of_members[WIB_IDX] + 2, dd->minimum_out_degrees, 
                               *dd->minimum_clique_size);
    min_in_deg = d_get_mindeg(wd.number_of_members[WIB_IDX] + 2, dd->minimum_in_degrees, 
                               *dd->minimum_clique_size);

    // initialize vertex order map
    for(int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE){
        dd->vertex_order_map[warp_write + ld.vertices[i].vertexid] = i;
    }
    __syncwarp();

    // ADD ONE VERTEX
    pvertexid = ld.vertices[wd.number_of_members[WIB_IDX]].vertexid;

    if (LANE_IDX == 0) {
        ld.vertices[wd.number_of_members[WIB_IDX]].label = 1;
        wd.number_of_members[WIB_IDX]++;
        wd.number_of_candidates[WIB_IDX]--;
    }

    // update degrees of adjacent vertices
    pneighbors_start = dd->out_offsets[pvertexid];
    pneighbors_end = dd->out_offsets[pvertexid + 1];

    for (uint64_t i = pneighbors_start + LANE_IDX; i < pneighbors_end; i += WARP_SIZE) {

        phelper1 = dd->vertex_order_map[warp_write + dd->out_neighbors[i]];

        if (phelper1 > -1) {
            ld.vertices[phelper1].in_mem_deg++;
            ld.vertices[phelper1].in_can_deg--;
        }
    }

    pneighbors_start = dd->in_offsets[pvertexid];
    pneighbors_end = dd->in_offsets[pvertexid + 1];

    for (uint64_t i = pneighbors_start + LANE_IDX; i < pneighbors_end; i += WARP_SIZE) {

        phelper1 = dd->vertex_order_map[warp_write + dd->in_neighbors[i]];

        if (phelper1 > -1) {
            ld.vertices[phelper1].out_mem_deg++;
            ld.vertices[phelper1].out_can_deg--;
        }
    }
    __syncwarp();

    // DIAMETER PRUNING
    d_diameter_pruning(dd, wd, ld, pvertexid, min_out_deg, min_in_deg);

    // DEGREE BASED PRUNING
    // sets success to false if failed found else leaves as true
    d_degree_pruning(dd, wd, ld);
}

// sets success as 2 if critical fail, 0 if failed found or invalid bound, 1 otherwise
__device__ void d_critical_vertex_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    int phelper1;                   // intersection
    int number_of_crit;         // pruning
    int warp_write;
    int pvertexid;                  // intersection
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;

    warp_write = WARP_IDX * *dd->WVERTICES_SIZE;

    // initialize vertex order map
    for(int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE){
        dd->vertex_order_map[warp_write + ld.vertices[i].vertexid] = i;
    }
    __syncwarp();

    // CRITICAL VERTEX PRUNING 
    // iterate through all vertices in clique
    for(int i = 0; i < wd.number_of_members[WIB_IDX]; i++){

        pvertexid = ld.vertices[i].vertexid;

        // if they are a critical vertex in out direction
        if (ld.vertices[i].out_mem_deg + ld.vertices[i].out_can_deg == 
            dd->minimum_out_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]] 
            && ld.vertices[i].out_can_deg > 0) {

            // iterate through all neighbors
            pneighbors_start = dd->out_offsets[pvertexid];
            pneighbors_end = dd->out_offsets[pvertexid + 1];

            for (uint64_t j = pneighbors_start + LANE_IDX; j < pneighbors_end; j += WARP_SIZE) {

                phelper1 = dd->vertex_order_map[warp_write + dd->out_neighbors[j]];

                // if neighbor is cand
                if (phelper1 >= wd.number_of_members[WIB_IDX]) {
                    ld.vertices[phelper1].label = 4;
                }
            }
        }

        // if they are a critical vertex in in direction
        if (ld.vertices[i].in_mem_deg + ld.vertices[i].in_can_deg == 
            dd->minimum_in_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]] 
            && ld.vertices[i].in_can_deg > 0) {

            // iterate through all neighbors
            pneighbors_start = dd->in_offsets[pvertexid];
            pneighbors_end = dd->in_offsets[pvertexid + 1];

            for (uint64_t j = pneighbors_start + LANE_IDX; j < pneighbors_end; j += WARP_SIZE) {

                phelper1 = dd->vertex_order_map[warp_write + dd->in_neighbors[j]];

                // if neighbor is cand
                if (phelper1 >= wd.number_of_members[WIB_IDX]) {
                    ld.vertices[phelper1].label = 4;
                }
            }
        }
    }
    __syncwarp();

    // reset vertex order map
    for(int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE){
        dd->vertex_order_map[warp_write + ld.vertices[i].vertexid] = -1;
    }
    __syncwarp();

    // sort vertices so that critical vertex adjacent candidates are immediately after vertices 
    // within the clique
    d_oe_sort_vert(ld.vertices + wd.number_of_members[WIB_IDX], wd.number_of_candidates[WIB_IDX], 
                   d_comp_vert_cv);

    // count number of critical adjacent vertices
    number_of_crit = 0;
    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += 
         WARP_SIZE) {

        if (ld.vertices[i].label == 4) {
            number_of_crit++;
        }
        else {
            break;
        }
    }
    // get sum
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        number_of_crit += __shfl_xor_sync(0xFFFFFFFF, number_of_crit, i);
    }

    // no crit found, nothing to be done, return
    if(number_of_crit == 0){
        return;
    }

    // initialize vertex order map and reset adjacencies
    // adjacencies[4] = 10, means vertex at position 4 in vertices is adjacent to 10 cv
    for(int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE){
        dd->adjacencies[warp_write + i] = 0;
        dd->vertex_order_map[warp_write + ld.vertices[i].vertexid] = i;
    }
    __syncwarp();

    // calculate adj_counters, adjacencies to critical vertices
    for (int i = wd.number_of_members[WIB_IDX]; i < wd.number_of_members[WIB_IDX] + 
        number_of_crit; i++) {

        pvertexid = ld.vertices[i].vertexid;

        // track 2hop adj
        pneighbors_start = dd->twohop_offsets[pvertexid];
        pneighbors_end = dd->twohop_offsets[pvertexid + 1];

        for (uint64_t j = pneighbors_start + LANE_IDX; j < pneighbors_end; j += WARP_SIZE) {

            phelper1 = dd->vertex_order_map[warp_write + dd->twohop_neighbors[j]];

            if (phelper1 > -1) {
                dd->adjacencies[warp_write + phelper1]++;
            }
        }
        __syncwarp();
    }

    // check for critical fails
    // all vertices within the clique must be within 2hops of the newly added critical vertex adj 
    // vertices
    for (int i = LANE_IDX; i < wd.number_of_members[WIB_IDX] && wd.success[WIB_IDX] == 1; i += 
        WARP_SIZE) {
        
        if (dd->adjacencies[warp_write + i] != number_of_crit) {
            wd.success[WIB_IDX] = 2;
            break;
        }
    }
    __syncwarp();

    if (wd.success[WIB_IDX] == 2) {
        // reset vertex order map
        for(int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE){
            dd->vertex_order_map[warp_write + ld.vertices[i].vertexid] = -1;
        }
        return;
    }

    // all critical adj vertices must all be within 2 hops of each other
    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.number_of_members[WIB_IDX] + 
        number_of_crit && wd.success[WIB_IDX] == 1; i += WARP_SIZE) {

        if (dd->adjacencies[warp_write + i] < number_of_crit - 1) {
            wd.success[WIB_IDX] = 2;
            break;
        }
    }
    __syncwarp();

    if (wd.success[WIB_IDX] == 2) {
        // reset vertex order map
        for(int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE){
            dd->vertex_order_map[warp_write + ld.vertices[i].vertexid] = -1;
        }
        return;
    }

    // update degrees of vertics next adjacent to cv
    for (int i = wd.number_of_members[WIB_IDX]; i < wd.number_of_members[WIB_IDX] + 
        number_of_crit; i++) {

        pvertexid = ld.vertices[i].vertexid;

        // update 1hop adj
        pneighbors_start = dd->out_offsets[pvertexid];
        pneighbors_end = dd->out_offsets[pvertexid + 1];

        for (uint64_t j = pneighbors_start + LANE_IDX; j < pneighbors_end; j += WARP_SIZE) {

            phelper1 = dd->vertex_order_map[warp_write + dd->out_neighbors[j]];

            if (phelper1 > -1) {
                ld.vertices[phelper1].in_mem_deg++;
                ld.vertices[phelper1].in_can_deg--;
            }
        }

        pneighbors_start = dd->in_offsets[pvertexid];
        pneighbors_end = dd->in_offsets[pvertexid + 1];

        for (uint64_t j = pneighbors_start + LANE_IDX; j < pneighbors_end; j += WARP_SIZE) {

            phelper1 = dd->vertex_order_map[warp_write + dd->in_neighbors[j]];

            if (phelper1 > -1) {
                ld.vertices[phelper1].out_mem_deg++;
                ld.vertices[phelper1].out_can_deg--;
            }
        }
        __syncwarp();
    }

    // no failed vertices found so add all critical vertex adj candidates to clique
    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.number_of_members[WIB_IDX] + 
            number_of_crit; i += WARP_SIZE) {

        ld.vertices[i].label = 1;
    }

    if (LANE_IDX == 0) {
        wd.number_of_members[WIB_IDX] += number_of_crit;
        wd.number_of_candidates[WIB_IDX] -= number_of_crit;
    }
    __syncwarp();

    // DIAMTER PRUNING
    d_diameter_pruning_cv(dd, wd, ld, number_of_crit);

    // DEGREE BASED PRUNING
    // sets success to false if failed found else leaves as true
    d_degree_pruning(dd, wd, ld);
}

// diameter pruning intitializes vertices labels and candidate indegs array for use in iterative 
// degree pruning
__device__ void d_diameter_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld, int pvertexid, 
                                   int min_out_deg, int min_in_deg)
{
    uint64_t lane_write;
    int phelper1;                       // intersection
    int phelper2;
    int lane_remaining_count;           // vertex iteration
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    uint64_t warp_write;

    warp_write = WARP_IDX * *dd->WVERTICES_SIZE;
    lane_write = warp_write + ((*dd->WVERTICES_SIZE / WARP_SIZE) * LANE_IDX);

    lane_remaining_count = 0;

    // set all candidates as invalid
    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; 
         i += WARP_SIZE) {
        
        ld.vertices[i].label = -1;
    }
    __syncwarp();

    // mark all candidates within two hops of added vertex as valid
    pneighbors_start = dd->twohop_offsets[pvertexid];
    pneighbors_end = dd->twohop_offsets[pvertexid + 1];

    for (uint64_t i = pneighbors_start + LANE_IDX; i < pneighbors_end; i += WARP_SIZE) {

        phelper1 = dd->vertex_order_map[warp_write + dd->twohop_neighbors[i]];

        if (phelper1 >= wd.number_of_members[WIB_IDX]) {

            // DEBUG - can this go inside if?
            ld.vertices[phelper1].label = 0;

            // only track mem degs of candidates which pass basic degree pruning
            if(ld.vertices[phelper1].out_mem_deg + ld.vertices[phelper1].out_can_deg >= min_out_deg
               && ld.vertices[phelper1].in_mem_deg + ld.vertices[phelper1].in_can_deg >= 
               min_in_deg){
                
                dd->lane_candidate_out_mem_degs[lane_write + lane_remaining_count] = 
                    ld.vertices[phelper1].out_mem_deg;
                dd->lane_candidate_in_mem_degs[lane_write + lane_remaining_count] = 
                    ld.vertices[phelper1].in_mem_deg;
                lane_remaining_count++;
            }
        }
    }
    __syncwarp();

    //  the following section combines the lane mem degs arrays into one warp array
    // scan to calculate write postion in warp arrays
    phelper2 = lane_remaining_count;
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
        if (LANE_IDX >= i) {
            lane_remaining_count += phelper1;
        }
        __syncwarp();
    }
    // lane remaining count sum is scan for last lane and its value
    if (LANE_IDX == WARP_SIZE - 1) {
        wd.remaining_count[WIB_IDX] = lane_remaining_count;
    }
    // make scan exclusive
    lane_remaining_count -= phelper2;

    // parallel write lane arrays to warp array
    for (int i = 0; i < phelper2; i++) {
        dd->candidate_out_mem_degs[warp_write + lane_remaining_count + i] = 
            dd->lane_candidate_out_mem_degs[lane_write + i];
        dd->candidate_in_mem_degs[warp_write + lane_remaining_count + i] = 
            dd->lane_candidate_in_mem_degs[lane_write + i];
    }
    __syncwarp();
}

__device__ void d_diameter_pruning_cv(GPU_Data* dd, Warp_Data& wd, Local_Data& ld, 
                                      int number_of_crit)
{
    uint64_t lane_write;
    int lane_remaining_count;           // vertex iteration
    int phelper1;                       // intersection
    int phelper2;
    uint64_t warp_write;

    warp_write = WARP_IDX * *dd->WVERTICES_SIZE;
    lane_write = warp_write + (*dd->WVERTICES_SIZE / WARP_SIZE) * LANE_IDX;
    lane_remaining_count = 0;

    // remove all cands who are not within 2hops of all newly added cands
    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += 
         WARP_SIZE) {

        if (dd->adjacencies[warp_write + i] == number_of_crit) {

            dd->lane_candidate_out_mem_degs[lane_write + lane_remaining_count] = 
                ld.vertices[i].out_mem_deg;
            dd->lane_candidate_in_mem_degs[lane_write + lane_remaining_count] = 
                ld.vertices[i].in_mem_deg;
            lane_remaining_count++;
        }
        else {
            ld.vertices[i].label = -1;
        }
    }

    // scan to calculate write postion in warp arrays
    phelper2 = lane_remaining_count;
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
        if (LANE_IDX >= i) {
            lane_remaining_count += phelper1;
        }
        __syncwarp();
    }
    // lane remaining count sum is scan for last lane and its value
    if (LANE_IDX == WARP_SIZE - 1) {
        wd.remaining_count[WIB_IDX] = lane_remaining_count;
    }
    // make scan exclusive
    lane_remaining_count -= phelper2;

    // parallel write lane arrays to warp array
    for (int i = 0; i < phelper2; i++) {
        dd->candidate_out_mem_degs[warp_write + lane_remaining_count + i] = 
            dd->lane_candidate_out_mem_degs[lane_write + i];
        dd->candidate_in_mem_degs[warp_write + lane_remaining_count + i] = 
            dd->lane_candidate_in_mem_degs[lane_write + i];
    }
    __syncwarp();
}

// sets success to false if failed or invalid bounds found else leaves as true
__device__ void d_degree_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    int lane_write;                 // place each lane will write in warp array
    int pvertexid;                  // helper variables
    int phelper1;
    int phelper2;
    int phelper3;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int lane_remaining_count;       // counter for lane intersection results
    int lane_removed_count;
    int warp_write;

    warp_write = WARP_IDX * *dd->WVERTICES_SIZE;
    lane_write = warp_write + ((*dd->WVERTICES_SIZE / WARP_SIZE) * LANE_IDX);

    // used for bound calculation
    d_oe_sort_int(dd->candidate_out_mem_degs + warp_write, wd.remaining_count[WIB_IDX], 
                  d_comp_int_desc);
    d_oe_sort_int(dd->candidate_in_mem_degs + warp_write, wd.remaining_count[WIB_IDX], 
                  d_comp_int_desc);

    //set bounds and min ext degs
    d_calculate_LU_bounds(dd, wd, ld, wd.remaining_count[WIB_IDX]);
    __syncwarp();

    // check whether bounds are valid
    if(wd.success[WIB_IDX] == false){
        // reset vertex order map
        for(int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE){
            dd->vertex_order_map[warp_write + ld.vertices[i].vertexid] = -1;
        }
        return;
    }

    // check for failed vertices
    for(int i = LANE_IDX; i < wd.number_of_members[WIB_IDX] && wd.success[WIB_IDX]; i += 
        WARP_SIZE){

        if(!d_vert_isextendable(ld.vertices[i], dd, wd, ld)){
            wd.success[WIB_IDX] = false;
            break;
        }
    }
    __syncwarp();

    // reset vertex order map
    if(!wd.success[WIB_IDX]){
        for(int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE){
            dd->vertex_order_map[warp_write + ld.vertices[i].vertexid] = -1;
        }
        return;
    }   

    if (LANE_IDX == 0) {
        wd.remaining_count[WIB_IDX] = 0;
        wd.removed_count[WIB_IDX] = 0;
    }
    lane_remaining_count = 0;
    lane_removed_count = 0;
    
    // check for invalid candidates
    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += 
         WARP_SIZE) {
        
        if (ld.vertices[i].label == 0 && d_cand_isvalid(ld.vertices[i], dd, wd, ld)) {
            dd->lane_remaining_candidates[lane_write + lane_remaining_count++] = i;
        }
        else {
            dd->lane_removed_candidates[lane_write + lane_removed_count++] = i;
        }
    }
    __syncwarp();

    // scan to calculate write postion in warp arrays
    phelper2 = lane_remaining_count;
    pvertexid = lane_removed_count;
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
        phelper3 = __shfl_up_sync(0xFFFFFFFF, lane_removed_count, i, WARP_SIZE);
        if (LANE_IDX >= i) {
            lane_remaining_count += phelper1;
            lane_removed_count += phelper3;
        }
        __syncwarp();
    }
    // lane remaining count sum is scan for last lane and its value
    if (LANE_IDX == WARP_SIZE - 1) {
        wd.remaining_count[WIB_IDX] = lane_remaining_count;
        wd.removed_count[WIB_IDX] = lane_removed_count;
    }
    __syncwarp();
    // make scan exclusive
    lane_remaining_count -= phelper2;
    lane_removed_count -= pvertexid;

    // parallel write lane arrays to warp array
    for (int i = 0; i < phelper2; i++) {
        dd->remaining_candidates[warp_write + lane_remaining_count + i] = 
            dd->lane_remaining_candidates[lane_write + i];
    }
    if (wd.remaining_count[WIB_IDX] >= wd.removed_count[WIB_IDX]){
        for (int i = 0; i < pvertexid; i++) {
            dd->removed_candidates[warp_write + lane_removed_count + i] = 
                dd->lane_removed_candidates[lane_write + i];
        }
    }
    __syncwarp();
    
    while (wd.remaining_count[WIB_IDX] > 0 && wd.removed_count[WIB_IDX] > 0) {

        // update degrees
        if (wd.remaining_count[WIB_IDX] < wd.removed_count[WIB_IDX]) {
            
            // via remaining, reset exdegs
            for (int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
                ld.vertices[i].in_can_deg = 0;
                ld.vertices[i].out_can_deg = 0;
            }
            __syncwarp();

            for(int i = 0; i < wd.remaining_count[WIB_IDX]; i++){

                pvertexid = ld.vertices[dd->remaining_candidates[warp_write + i]].vertexid;

                // update degrees of remaining adjacent vertices
                pneighbors_start = dd->out_offsets[pvertexid];
                pneighbors_end = dd->out_offsets[pvertexid + 1];

                for (uint64_t j = pneighbors_start + LANE_IDX; j < pneighbors_end; j += WARP_SIZE) {

                    phelper1 = dd->vertex_order_map[warp_write + dd->out_neighbors[j]];

                    if (phelper1 > -1) {
                        ld.vertices[phelper1].in_can_deg++;
                    }
                }

                pneighbors_start = dd->in_offsets[pvertexid];
                pneighbors_end = dd->in_offsets[pvertexid + 1];

                for (uint64_t j = pneighbors_start + LANE_IDX; j < pneighbors_end; j += WARP_SIZE) {

                    phelper1 = dd->vertex_order_map[warp_write + dd->in_neighbors[j]];

                    if (phelper1 > -1) {
                        ld.vertices[phelper1].out_can_deg++;
                    }
                }
                __syncwarp();
            }
        }
        else {
            
            for(int i = 0; i < wd.removed_count[WIB_IDX]; i++){

                pvertexid = ld.vertices[dd->removed_candidates[warp_write + i]].vertexid;

                // update degrees of remaining adjacent vertices
                pneighbors_start = dd->out_offsets[pvertexid];
                pneighbors_end = dd->out_offsets[pvertexid + 1];

                for (uint64_t j = pneighbors_start + LANE_IDX; j < pneighbors_end; j += WARP_SIZE) {

                    phelper1 = dd->vertex_order_map[warp_write + dd->out_neighbors[j]];

                    if (phelper1 > -1) {
                        ld.vertices[phelper1].in_can_deg--;
                    }
                }

                pneighbors_start = dd->in_offsets[pvertexid];
                pneighbors_end = dd->in_offsets[pvertexid + 1];

                for (uint64_t j = pneighbors_start + LANE_IDX; j < pneighbors_end; j += WARP_SIZE) {

                    phelper1 = dd->vertex_order_map[warp_write + dd->in_neighbors[j]];

                    if (phelper1 > -1) {
                        ld.vertices[phelper1].out_can_deg--;
                    }
                }
                __syncwarp();
            }
        }

        lane_remaining_count = 0;

        for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
            if (d_cand_isvalid(ld.vertices[dd->remaining_candidates[warp_write + i]], dd, wd, ld)) {
                
                dd->lane_candidate_out_mem_degs[lane_write + lane_remaining_count] = 
                    ld.vertices[dd->remaining_candidates[warp_write + i]].out_mem_deg;
                dd->lane_candidate_in_mem_degs[lane_write + lane_remaining_count] = 
                    ld.vertices[dd->remaining_candidates[warp_write + i]].in_mem_deg;
                lane_remaining_count++;
            }
        }
        __syncwarp();

        // scan to calculate write postion in warp arrays
        phelper2 = lane_remaining_count;
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
            if (LANE_IDX >= i) {
                lane_remaining_count += phelper1;
            }
            __syncwarp();
        }
        // lane remaining count sum is scan for last lane and its value
        if (LANE_IDX == WARP_SIZE - 1) {
            wd.num_val_cands[WIB_IDX] = lane_remaining_count;
        }
        // make scan exclusive
        lane_remaining_count -= phelper2;

        // parallel write lane arrays to warp array
        for (int i = 0; i < phelper2; i++) {
            dd->candidate_out_mem_degs[warp_write + lane_remaining_count + i] = 
                dd->lane_candidate_out_mem_degs[lane_write + i];
            dd->candidate_in_mem_degs[warp_write + lane_remaining_count + i] = 
                dd->lane_candidate_in_mem_degs[lane_write + i];
        }
        __syncwarp();

        d_oe_sort_int(dd->candidate_out_mem_degs + warp_write, wd.num_val_cands[WIB_IDX], 
                      d_comp_int_desc);
        d_oe_sort_int(dd->candidate_in_mem_degs + warp_write, wd.num_val_cands[WIB_IDX], 
                      d_comp_int_desc);

        // set bounds and min ext degs
        d_calculate_LU_bounds(dd, wd, ld, wd.num_val_cands[WIB_IDX]);
        __syncwarp();

        // check whether bounds are valid
        if(wd.success[WIB_IDX] == false){
            // reset vertex order map
            for(int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE){
                dd->vertex_order_map[warp_write + ld.vertices[i].vertexid] = -1;
            }
            return;
        }

        // check for failed vertices
        for(int i = LANE_IDX; i < wd.number_of_members[WIB_IDX] && wd.success[WIB_IDX]; i += 
            WARP_SIZE){

            if(!d_vert_isextendable(ld.vertices[i], dd, wd, ld)){
                wd.success[WIB_IDX] = false;
                break;
            }
        }
        __syncwarp();

        // reset vertex order map
        if(!wd.success[WIB_IDX]){
            for(int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE){
                dd->vertex_order_map[warp_write + ld.vertices[i].vertexid] = -1;
            }
            return;
        }   

        lane_remaining_count = 0;
        lane_removed_count = 0;

        // check for failed candidates
        for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
            if (d_cand_isvalid(ld.vertices[dd->remaining_candidates[warp_write + i]], dd, wd, ld)) {
                dd->lane_remaining_candidates[lane_write + lane_remaining_count++] = 
                    dd->remaining_candidates[warp_write + i];
            }
            else {
                dd->lane_removed_candidates[lane_write + lane_removed_count++] = 
                    dd->remaining_candidates[warp_write + i];
            }
        }
        __syncwarp();

        // scan to calculate write postion in warp arrays
        phelper2 = lane_remaining_count;
        pvertexid = lane_removed_count;
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
            phelper3 = __shfl_up_sync(0xFFFFFFFF, lane_removed_count, i, WARP_SIZE);
            if (LANE_IDX >= i) {
                lane_remaining_count += phelper1;
                lane_removed_count += phelper3;
            }
            __syncwarp();
        }
        // lane remaining count sum is scan for last lane and its value
        if (LANE_IDX == WARP_SIZE - 1) {
            wd.num_val_cands[WIB_IDX] = lane_remaining_count;
            wd.removed_count[WIB_IDX] = lane_removed_count;
        }
        __syncwarp();
        // make scan exclusive
        lane_remaining_count -= phelper2;
        lane_removed_count -= pvertexid;

        // parallel write lane arrays to warp array
        for (int i = 0; i < phelper2; i++) {
            dd->remaining_candidates[warp_write + lane_remaining_count + i] = 
                dd->lane_remaining_candidates[lane_write + i];
        }
        if (wd.num_val_cands[WIB_IDX] >= wd.removed_count[WIB_IDX]){
            for (int i = 0; i < pvertexid; i++) {
                dd->removed_candidates[warp_write + lane_removed_count + i] = 
                    dd->lane_removed_candidates[lane_write + i];
            }
        }

        if (LANE_IDX == 0) {
            wd.remaining_count[WIB_IDX] = wd.num_val_cands[WIB_IDX];
        }
        __syncwarp();
    }

    // reset vertex order map before condensing
    for(int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE){
        dd->vertex_order_map[warp_write + ld.vertices[i].vertexid] = -1;
    }

    // condense vertices out of place
    for(int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE){
        dd->temp_vertex_array[warp_write + i] = 
            ld.vertices[dd->remaining_candidates[warp_write + i]];
    }
    __syncwarp();
    for(int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE){
        ld.vertices[wd.number_of_members[WIB_IDX] + i] = dd->temp_vertex_array[warp_write + i];
    }

    if (LANE_IDX == 0) {
        wd.total_vertices[WIB_IDX] = wd.total_vertices[WIB_IDX] - wd.number_of_candidates[WIB_IDX] 
            + wd.remaining_count[WIB_IDX];

        wd.number_of_candidates[WIB_IDX] = wd.remaining_count[WIB_IDX];
    }
    __syncwarp();
}

__device__ void d_calculate_LU_bounds(GPU_Data* dd, Warp_Data& wd, Local_Data& ld, 
                                      int number_of_candidates)
{
    // TODO - parallelize some of the bound calculation
    if(LANE_IDX == 0){
        //lower & upper bound are initialized using the degree of vertex in S
        //and tighten using the degree of vertex in ext_S
        int i, ntightened_max_cands;
        int warp_write;

        warp_write = WARP_IDX * *dd->WVERTICES_SIZE;

        //clq_clqdeg means: v in S (clq) 's indegree (clqdeg)
        wd.nmin_clq_clqdeg_o[WIB_IDX] = ld.vertices[0].out_mem_deg;
        wd.nminclqdeg_candeg_o[WIB_IDX] = ld.vertices[0].out_can_deg;
        wd.nclq_clqdeg_sum_o[WIB_IDX] = ld.vertices[0].out_mem_deg;
        wd.nmin_clq_totaldeg_o[WIB_IDX] = ld.vertices[0].out_mem_deg+ld.vertices[0].out_can_deg;

        wd.nmin_clq_clqdeg_i[WIB_IDX] = ld.vertices[0].in_mem_deg;
        wd.nminclqdeg_candeg_i[WIB_IDX] = ld.vertices[0].in_can_deg;
        wd.nclq_clqdeg_sum_i[WIB_IDX] = ld.vertices[0].in_mem_deg;
        wd.nmin_clq_totaldeg_i[WIB_IDX] = ld.vertices[0].in_mem_deg+ld.vertices[0].in_can_deg;

        for(i=1;i<wd.number_of_members[WIB_IDX];i++)
        {
            // out direction
            wd.nclq_clqdeg_sum_o[WIB_IDX] += ld.vertices[i].out_mem_deg;
            if(wd.nmin_clq_clqdeg_o[WIB_IDX]>ld.vertices[i].out_mem_deg)
            {
                wd.nmin_clq_clqdeg_o[WIB_IDX] = ld.vertices[i].out_mem_deg;
                wd.nminclqdeg_candeg_o[WIB_IDX] = ld.vertices[i].out_can_deg;
            }
            else if(wd.nmin_clq_clqdeg_o[WIB_IDX]==ld.vertices[i].out_mem_deg)
            {
                if(wd.nminclqdeg_candeg_o[WIB_IDX]>ld.vertices[i].out_can_deg)
                    wd.nminclqdeg_candeg_o[WIB_IDX] = ld.vertices[i].out_can_deg;
            }

            if(wd.nmin_clq_totaldeg_o[WIB_IDX]>ld.vertices[i].out_mem_deg+ld.vertices[i].out_can_deg){
                wd.nmin_clq_totaldeg_o[WIB_IDX] = ld.vertices[i].out_mem_deg+ld.vertices[i].out_can_deg;
            }

            // in direction
            wd.nclq_clqdeg_sum_i[WIB_IDX] += ld.vertices[i].in_mem_deg;
            if(wd.nmin_clq_clqdeg_i[WIB_IDX]>ld.vertices[i].in_mem_deg)
            {
                wd.nmin_clq_clqdeg_i[WIB_IDX] = ld.vertices[i].in_mem_deg;
                wd.nminclqdeg_candeg_i[WIB_IDX] = ld.vertices[i].in_can_deg;
            }
            else if(wd.nmin_clq_clqdeg_i[WIB_IDX]==ld.vertices[i].in_mem_deg)
            {
                if(wd.nminclqdeg_candeg_i[WIB_IDX]>ld.vertices[i].in_can_deg){
                    wd.nminclqdeg_candeg_i[WIB_IDX] = ld.vertices[i].in_can_deg;
                }
            }

            if(wd.nmin_clq_totaldeg_i[WIB_IDX]>ld.vertices[i].in_mem_deg+ld.vertices[i].in_can_deg){
                wd.nmin_clq_totaldeg_i[WIB_IDX] = ld.vertices[i].in_mem_deg+ld.vertices[i].in_can_deg;
            }
        }

        wd.min_ext_out_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX]+1, dd->minimum_out_degrees, (*dd->minimum_clique_size));
        wd.min_ext_in_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX]+1, dd->minimum_in_degrees, (*dd->minimum_clique_size));
        
        if(wd.nmin_clq_clqdeg_o[WIB_IDX]<dd->minimum_out_degrees[wd.number_of_members[WIB_IDX]] || wd.nmin_clq_clqdeg_i[WIB_IDX]<dd->minimum_in_degrees[wd.number_of_members[WIB_IDX]])//check the requirment of bound pruning rule
        {
            // ==== calculate L_min and U_min ====
            //initialize lower bound
            int nmin_cands = max((d_get_mindeg(wd.number_of_members[WIB_IDX], dd->minimum_out_degrees, (*dd->minimum_clique_size))-wd.nmin_clq_clqdeg_o[WIB_IDX]),
                    (d_get_mindeg(wd.number_of_members[WIB_IDX], dd->minimum_in_degrees, (*dd->minimum_clique_size))-wd.nmin_clq_clqdeg_i[WIB_IDX]));
            int nmin_cands_o = nmin_cands;

            while(nmin_cands_o<=wd.nminclqdeg_candeg_o[WIB_IDX] && wd.nmin_clq_clqdeg_o[WIB_IDX]+nmin_cands_o<dd->minimum_out_degrees[wd.number_of_members[WIB_IDX]+nmin_cands_o]){
                nmin_cands_o++;
            }

            if(wd.nmin_clq_clqdeg_o[WIB_IDX]+nmin_cands_o<dd->minimum_out_degrees[wd.number_of_members[WIB_IDX]+nmin_cands_o]){
                wd.success[WIB_IDX] = false;
                return;
            }

            int nmin_cands_i = nmin_cands;

            while(nmin_cands_i<=wd.nminclqdeg_candeg_i[WIB_IDX] && wd.nmin_clq_clqdeg_i[WIB_IDX]+nmin_cands_i<dd->minimum_in_degrees[wd.number_of_members[WIB_IDX]+nmin_cands_i]){
                nmin_cands_i++;
            }

            if(wd.nmin_clq_clqdeg_i[WIB_IDX]+nmin_cands_i<dd->minimum_in_degrees[wd.number_of_members[WIB_IDX]+nmin_cands_i]){
                wd.success[WIB_IDX] = false;
                return;
            }

            wd.lower_bound[WIB_IDX] = max(nmin_cands_o, nmin_cands_i);

            //initialize upper bound
            wd.upper_bound[WIB_IDX] = min((int)(wd.nmin_clq_totaldeg_o[WIB_IDX]/(*dd->minimum_out_degree_ratio)),
                    (int)(wd.nmin_clq_totaldeg_i[WIB_IDX]/(*dd->minimum_in_degree_ratio)))+1-wd.number_of_members[WIB_IDX];
            
            if(wd.upper_bound[WIB_IDX]>number_of_candidates){
                wd.upper_bound[WIB_IDX] = number_of_candidates;
            }

            // ==== tighten lower bound and upper bound based on the clique degree of candidates ====
            if(wd.lower_bound[WIB_IDX]<wd.upper_bound[WIB_IDX])
            {
                //tighten lower bound
                wd.ncand_clqdeg_sum_o[WIB_IDX] = 0;
                wd.ncand_clqdeg_sum_i[WIB_IDX] = 0;

                for(i=0;i<wd.lower_bound[WIB_IDX];i++)
                {
                    wd.ncand_clqdeg_sum_o[WIB_IDX] += dd->candidate_out_mem_degs[warp_write + i];
                    wd.ncand_clqdeg_sum_i[WIB_IDX] += dd->candidate_in_mem_degs[warp_write + i];
                }

                while(i<wd.upper_bound[WIB_IDX]
                        && wd.nclq_clqdeg_sum_o[WIB_IDX]+wd.ncand_clqdeg_sum_i[WIB_IDX]<wd.number_of_members[WIB_IDX]*dd->minimum_out_degrees[wd.number_of_members[WIB_IDX]+i]
                        && wd.nclq_clqdeg_sum_i[WIB_IDX]+wd.ncand_clqdeg_sum_o[WIB_IDX]<wd.number_of_members[WIB_IDX]*dd->minimum_in_degrees[wd.number_of_members[WIB_IDX]+i])
                {
                    wd.ncand_clqdeg_sum_o[WIB_IDX] += dd->candidate_out_mem_degs[warp_write + i];
                    wd.ncand_clqdeg_sum_i[WIB_IDX] += dd->candidate_in_mem_degs[warp_write + i];
                    i++;
                }

                if(wd.nclq_clqdeg_sum_o[WIB_IDX]+wd.ncand_clqdeg_sum_o[WIB_IDX]<wd.number_of_members[WIB_IDX]*dd->minimum_out_degrees[wd.number_of_members[WIB_IDX]+i]
                    && wd.nclq_clqdeg_sum_i[WIB_IDX]+wd.ncand_clqdeg_sum_i[WIB_IDX]<wd.number_of_members[WIB_IDX]*dd->minimum_in_degrees[wd.number_of_members[WIB_IDX]+i]){
                    wd.success[WIB_IDX] = false;
                    return;
                }
                else //tighten upper bound
                {
                    wd.lower_bound[WIB_IDX] = i;

                    ntightened_max_cands = i;
                    while(i<wd.upper_bound[WIB_IDX])
                    {
                        wd.ncand_clqdeg_sum_o[WIB_IDX] += dd->candidate_out_mem_degs[warp_write + i];
                        wd.ncand_clqdeg_sum_i[WIB_IDX] += dd->candidate_in_mem_degs[warp_write + i];
                        i++;
                        if(wd.nclq_clqdeg_sum_o[WIB_IDX]+wd.ncand_clqdeg_sum_i[WIB_IDX]>=wd.number_of_members[WIB_IDX]*dd->minimum_out_degrees[wd.number_of_members[WIB_IDX]+i]
                            && wd.nclq_clqdeg_sum_i[WIB_IDX]+wd.ncand_clqdeg_sum_o[WIB_IDX]>=wd.number_of_members[WIB_IDX]*dd->minimum_in_degrees[wd.number_of_members[WIB_IDX]+i]){
                            ntightened_max_cands = i;
                        }
                    }
                    if(wd.upper_bound[WIB_IDX]>ntightened_max_cands){
                        wd.upper_bound[WIB_IDX] = ntightened_max_cands;
                    }

                    if(wd.lower_bound[WIB_IDX]>1)
                    {
                        wd.min_ext_out_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX]+wd.lower_bound[WIB_IDX], dd->minimum_out_degrees, (*dd->minimum_clique_size));
                        wd.min_ext_in_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX]+wd.lower_bound[WIB_IDX], dd->minimum_in_degrees, (*dd->minimum_clique_size));
                    }
                }
            }
        }
        else
        {
            wd.upper_bound[WIB_IDX] = number_of_candidates;

            if(wd.number_of_members[WIB_IDX]<(*dd->minimum_clique_size)){
                wd.lower_bound[WIB_IDX] = (*dd->minimum_clique_size)-wd.number_of_members[WIB_IDX];
            }
            else{
                wd.lower_bound[WIB_IDX] = 0;
            }
        }

        if(wd.number_of_members[WIB_IDX]+wd.upper_bound[WIB_IDX]<(*dd->minimum_clique_size)){
            wd.success[WIB_IDX] = false;
            return;
        }

        if(wd.upper_bound[WIB_IDX] < 0 || wd.upper_bound[WIB_IDX] < wd.lower_bound[WIB_IDX])
        {
            wd.success[WIB_IDX] = false;
            return;
        }
    }
}

__device__ void d_check_for_clique(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    uint64_t start_write;
    bool clique;
    int min_out_deg;
    int min_in_deg;

    if (wd.number_of_members[WIB_IDX] < *dd->minimum_clique_size) {
        return;
    }

    clique = true;

    min_out_deg = dd->minimum_out_degrees[wd.number_of_members[WIB_IDX]];
    min_in_deg = dd->minimum_in_degrees[wd.number_of_members[WIB_IDX]];

    for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX]; k += WARP_SIZE) {
        if (ld.vertices[k].out_mem_deg < min_out_deg || ld.vertices[k].in_mem_deg < min_in_deg) {
            clique = false;
            break;
        }
    }
    // set to false if any threads in warp do not meet degree requirement
    clique = !(__any_sync(0xFFFFFFFF, !clique));

    // if clique write to warp buffer for cliques
    if (clique) {
        start_write = (*dd->WCLIQUES_SIZE * WARP_IDX) + 
            dd->wcliques_offset[(*dd->WCLIQUES_OFFSET_SIZE * WARP_IDX) + 
            dd->wcliques_count[WARP_IDX]];

        for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX]; k += WARP_SIZE) {
            dd->wcliques_vertex[start_write + k] = ld.vertices[k].vertexid;
        }
        if (LANE_IDX == 0) {
            (dd->wcliques_count[WARP_IDX])++;

            dd->wcliques_offset[*dd->WCLIQUES_OFFSET_SIZE * WARP_IDX + 
                dd->wcliques_count[WARP_IDX]] = start_write - (*dd->WCLIQUES_SIZE * WARP_IDX) + 
                wd.number_of_members[WIB_IDX];
        }
    }
}

__device__ void d_write_to_tasks(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    uint64_t start_write;

    start_write = (*dd->WTASKS_SIZE * WARP_IDX) + dd->wtasks_offset[*dd->WTASKS_OFFSET_SIZE * WARP_IDX + dd->wtasks_count[WARP_IDX]];

    for (int k = LANE_IDX; k < wd.total_vertices[WIB_IDX]; k += WARP_SIZE) {
        dd->wtasks_vertices[start_write + k] = ld.vertices[k];
        dd->wtasks_vertices[start_write + k].lvl2adj = 0;
    }
    if (LANE_IDX == 0) {
        dd->wtasks_count[WARP_IDX]++;
        dd->wtasks_offset[(*dd->WTASKS_OFFSET_SIZE * WARP_IDX) + dd->wtasks_count[WARP_IDX]] = start_write - (*dd->WTASKS_SIZE * WARP_IDX) + wd.total_vertices[WIB_IDX];
    }
}

// --- TERTIARY KENERLS ---
// searches an int array for a certain int, returns the position in the array that item was found, 
// or -1 if not found
__device__ int d_b_search_int(int* search_array, int array_size, int search_number)
{
    // ALGO - BINARY
    // TYPE - SERIAL
    // SPEED - O(log(n))
    
    int low;
    int high;
    int mid;
    int mid_value;
    int comp;

    low = 0;
    high = array_size - 1;

    while (low < high) {
        mid = (low + high) / 2;
        mid_value = search_array[mid];
        comp = (mid_value < search_number);

        low = low + comp * (mid + 1 - low);
        high = high - !comp * (high - mid);
    }

    // Now low == high, check if it's the search_number
    return (search_array[low] == search_number) ? low : -1;
}

// consider using merge
__device__ void d_oe_sort_vert(Vertex* target, int size, int (*func)(Vertex&, Vertex&))
{
    // ALGO - ODD/EVEN
    // TYPE - PARALLEL
    // SPEED - O(n^2)

    Vertex vertex1;
    Vertex vertex2;

    for (int i = 0; i < size; i++) {
        for (int j = (i % 2) + (LANE_IDX * 2); j < size - 1; j += (WARP_SIZE * 2)) {
            vertex1 = target[j];
            vertex2 = target[j + 1];

            if (func(vertex1, vertex2) == 1) {
                target[j] = vertex2;
                target[j + 1] = vertex1;
            }
        }
        __syncwarp();
    }
}

__device__ void d_oe_sort_int(int* target, int size, int (*func)(int, int))
{
    // ALGO - ODD/EVEN
    // TYPE - PARALLEL
    // SPEED - O(n^2)

    int num1;
    int num2;

    for (int i = 0; i < size; i++) {
        for (int j = (i % 2) + (LANE_IDX * 2); j < size - 1; j += (WARP_SIZE * 2)) {
            num1 = target[j];
            num2 = target[j + 1];

            if (func(num1, num2) == 1) {
                target[j] = num2;
                target[j + 1] = num1;
            }
        }
        __syncwarp();
    }
}

// --- DEBUG KERNELS ---
__device__ void d_print_vertices(Vertex* vertices, int size)
{
    printf("\nOffsets:\n0 %i\nVertex:\n", size);
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].vertexid);
    }
    printf("\nLabel:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].label);
    }
    printf("\nOut-Mem-Deg:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].out_mem_deg);
    }
    printf("\nOut-Can-Deg:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].out_can_deg);
    }
    printf("\nIn-Mem-Deg:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].in_mem_deg);
    }
    printf("\nIn-Can-Deg:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].in_can_deg);
    }
    printf("\nLvl2adj:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].lvl2adj);
    }
    printf("\n");
}