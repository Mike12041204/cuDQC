#include "../inc/common.h"
#include "../inc/device_general.h"
#include "../inc/device_expansion.h"
#include "../inc/device_helper.h"
#include "../inc/device_debug.h"

__global__ void d_expand_level(GPU_Data dd)
{
    // data is stored in data structures to reduce the number of variables that need to be passed to methods
    __shared__ Warp_Data wd;
    Local_Data ld;

    // helper variables, not passed through to any methods
    int num_mem;
    int method_return;
    int index;



    /*
    * The program alternates between reading and writing between to 'tasks' arrays in device global memory. The program will read from one tasks, expand to the next level by generating and pruning, then it will write to the
    * other tasks array. It will write the first EXPAND_THRESHOLD to the tasks array and the rest to the top of the buffer. The buffers acts as a stack containing the excess data not being expanded from tasks. Since the 
    * buffer acts as a stack, in a last-in first-out manner, a subsection of the search space will be expanded until completion. This system allows the problem to essentially be divided into smaller problems and thus 
    * require less memory to handle.
    */
    if ((*(dd.current_level)) % 2 == 0) {
        dd.read_count = dd.tasks1_count;
        dd.read_offsets = dd.tasks1_offset;
        dd.read_vertices = dd.tasks1_vertices;
    }
    else {
        dd.read_count = dd.tasks2_count;
        dd.read_offsets = dd.tasks2_offset;
        dd.read_vertices = dd.tasks2_vertices;
    }



    // --- CURRENT LEVEL ---

    // scheduling toggle = 0, dynamic intersection
    if (*dd.scheduling_toggle == 0) {
        // initialize i for each warp
        int i = 0;
        if (LANE_IDX == 0) {
            i = atomicAdd(dd.current_task, 1);
        }
        i = __shfl_sync(0xFFFFFFFF, i, 0);

        while (i < (*(dd.read_count)))
        {
            // get information on vertices being handled within tasks
            if (LANE_IDX == 0) {
                wd.start[WIB_IDX] = dd.read_offsets[i];
                wd.end[WIB_IDX] = dd.read_offsets[i + 1];
                wd.tot_vert[WIB_IDX] = wd.end[WIB_IDX] - wd.start[WIB_IDX];
            }
            __syncwarp();

            // each warp gets partial number of members
            num_mem = 0;
            for (uint64_t j = wd.start[WIB_IDX] + LANE_IDX; j < wd.end[WIB_IDX]; j += WARP_SIZE) {
                if (dd.read_vertices[j].label != 1) {
                    break;
                }
                num_mem++;
            }
            // sum members across warp
            for (int k = 1; k < 32; k *= 2) {
                num_mem += __shfl_xor_sync(0xFFFFFFFF, num_mem, k);
            }

            if (LANE_IDX == 0) {
                wd.num_mem[WIB_IDX] = num_mem;
                wd.num_cand[WIB_IDX] = wd.tot_vert[WIB_IDX] - wd.num_mem[WIB_IDX];
                wd.expansions[WIB_IDX] = wd.num_cand[WIB_IDX];
            }
            __syncwarp();




            // LOOKAHEAD PRUNING
            method_return = d_lookahead_pruning(dd, wd, ld);
            if (method_return) {
                // schedule warps next task
                if (LANE_IDX == 0) {
                    i = atomicAdd(dd.current_task, 1);
                }
                i = __shfl_sync(0xFFFFFFFF, i, 0);
                continue;
            }



            // --- NEXT LEVEL ---
            for (int j = 0; j < wd.expansions[WIB_IDX]; j++)
            {



                // REMOVE ONE VERTEX
                if (j > 0) {
                    method_return = d_remove_one_vertex(dd, wd, ld);
                    if (method_return) {
                        break;
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
                    ld.vertices = dd.global_vertices + (WVERTICES_SIZE * WARP_IDX);
                }

                for (index = LANE_IDX; index < wd.number_of_members[WIB_IDX]; index += WARP_SIZE) {
                    ld.vertices[index] = dd.read_vertices[wd.start[WIB_IDX] + index];
                }
                for (; index < wd.total_vertices[WIB_IDX] - 1; index += WARP_SIZE) {
                    ld.vertices[index + 1] = dd.read_vertices[wd.start[WIB_IDX] + index];
                }

                if (LANE_IDX == 0) {
                    ld.vertices[wd.number_of_members[WIB_IDX]] = dd.read_vertices[wd.start[WIB_IDX] + wd.total_vertices[WIB_IDX] - 1];
                }
                __syncwarp();



                // ADD ONE VERTEX
                method_return = d_add_one_vertex(dd, wd, ld);

                // if failed found check for clique and continue on to the next iteration
                if (method_return == 1) {
                    if (wd.number_of_members[WIB_IDX] >= (*dd.minimum_clique_size)) {
                        d_check_for_clique(dd, wd, ld);
                    }
                    continue;
                }



                // CRITICAL VERTEX PRUNING
                method_return = d_critical_vertex_pruning(dd, wd, ld);

                // critical fail, cannot be clique continue onto next iteration
                if (method_return == 2) {
                    continue;
                }



                // HANDLE CLIQUES
                if (wd.number_of_members[WIB_IDX] >= (*dd.minimum_clique_size)) {
                    d_check_for_clique(dd, wd, ld);
                }

                // if vertex in x found as not extendable continue to next iteration
                if (method_return == 1) {
                    continue;
                }



                // WRITE TASKS TO BUFFERS
                // sort vertices in Quick efficient enumeration order before writing
                d_sort(ld.vertices, wd.total_vertices[WIB_IDX], d_sort_vert_Q);

                if (wd.number_of_candidates[WIB_IDX] > 0) {
                    d_write_to_tasks(dd, wd, ld);
                }
            }



            // schedule warps next task
            if (LANE_IDX == 0) {
                i = atomicAdd(dd.current_task, 1);
            }
            i = __shfl_sync(0xFFFFFFFF, i, 0);
        }
    }
    else {
        for (int i = WARP_IDX; i < (*(dd.read_count)); i += NUMBER_OF_WARPS)
        {
            // get information on vertices being handled within tasks
            if (LANE_IDX == 0) {
                wd.start[WIB_IDX] = dd.read_offsets[i];
                wd.end[WIB_IDX] = dd.read_offsets[i + 1];
                wd.tot_vert[WIB_IDX] = wd.end[WIB_IDX] - wd.start[WIB_IDX];
            }
            __syncwarp();

            // each warp gets partial number of members
            num_mem = 0;
            for (uint64_t j = wd.start[WIB_IDX] + LANE_IDX; j < wd.end[WIB_IDX]; j += WARP_SIZE) {
                if (dd.read_vertices[j].label != 1) {
                    break;
                }
                num_mem++;
            }
            // sum members across warp
            for (int k = 1; k < 32; k *= 2) {
                num_mem += __shfl_xor_sync(0xFFFFFFFF, num_mem, k);
            }

            if (LANE_IDX == 0) {
                wd.num_mem[WIB_IDX] = num_mem;
                wd.num_cand[WIB_IDX] = wd.tot_vert[WIB_IDX] - wd.num_mem[WIB_IDX];
                wd.expansions[WIB_IDX] = wd.num_cand[WIB_IDX];
            }
            __syncwarp();




            // LOOKAHEAD PRUNING
            method_return = d_lookahead_pruning(dd, wd, ld);
            if (method_return) {
                continue;
            }



            // --- NEXT LEVEL ---
            for (int j = 0; j < wd.expansions[WIB_IDX]; j++)
            {



                // REMOVE ONE VERTEX
                if (j > 0) {
                    method_return = d_remove_one_vertex(dd, wd, ld);
                    if (method_return) {
                        break;
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
                    ld.vertices = dd.global_vertices + (WVERTICES_SIZE * WARP_IDX);
                }

                for (index = LANE_IDX; index < wd.number_of_members[WIB_IDX]; index += WARP_SIZE) {
                    ld.vertices[index] = dd.read_vertices[wd.start[WIB_IDX] + index];
                }
                for (; index < wd.total_vertices[WIB_IDX] - 1; index += WARP_SIZE) {
                    ld.vertices[index + 1] = dd.read_vertices[wd.start[WIB_IDX] + index];
                }

                if (LANE_IDX == 0) {
                    ld.vertices[wd.number_of_members[WIB_IDX]] = dd.read_vertices[wd.start[WIB_IDX] + wd.total_vertices[WIB_IDX] - 1];
                }
                __syncwarp();



                // ADD ONE VERTEX
                method_return = d_add_one_vertex(dd, wd, ld);

                // if failed found check for clique and continue on to the next iteration
                if (method_return == 1) {
                    if (wd.number_of_members[WIB_IDX] >= (*dd.minimum_clique_size)) {
                        d_check_for_clique(dd, wd, ld);
                    }
                    continue;
                }



                // CRITICAL VERTEX PRUNING
                method_return = d_critical_vertex_pruning(dd, wd, ld);

                // critical fail, cannot be clique continue onto next iteration
                if (method_return == 2) {
                    continue;
                }



                // HANDLE CLIQUES
                if (wd.number_of_members[WIB_IDX] >= (*dd.minimum_clique_size)) {
                    d_check_for_clique(dd, wd, ld);
                }

                // if vertex in x found as not extendable continue to next iteration
                if (method_return == 1) {
                    continue;
                }



                // WRITE TASKS TO BUFFERS
                // sort vertices in Quick efficient enumeration order before writing
                d_sort(ld.vertices, wd.total_vertices[WIB_IDX], d_sort_vert_Q);

                if (wd.number_of_candidates[WIB_IDX] > 0) {
                    d_write_to_tasks(dd, wd, ld);
                }
            }
        }
    }



    if (LANE_IDX == 0) {
        // sum to find tasks count
        atomicAdd(dd.total_tasks, dd.wtasks_count[WARP_IDX]);
        atomicAdd(dd.total_cliques, dd.wcliques_count[WARP_IDX]);
    }

    if (IDX == 0) {
        (*(dd.buffer_offset_start)) = (*(dd.buffer_count)) + 1;
        (*(dd.buffer_start)) = dd.buffer_offset[(*(dd.buffer_count))];
        (*(dd.cliques_offset_start)) = (*(dd.cliques_count)) + 1;
        (*(dd.cliques_start)) = dd.cliques_offset[(*(dd.cliques_count))];
    }
}

__global__ void transfer_buffers(GPU_Data dd)
{
    __shared__ uint64_t tasks_write[WARPS_PER_BLOCK];
    __shared__ int tasks_offset_write[WARPS_PER_BLOCK];
    __shared__ uint64_t cliques_write[WARPS_PER_BLOCK];
    __shared__ int cliques_offset_write[WARPS_PER_BLOCK];

    __shared__ int twarp;
    __shared__ int toffsetwrite;
    __shared__ int twrite;
    __shared__ int tasks_end;

    if ((*(dd.current_level)) % 2 == 0) {
        dd.write_count = dd.tasks2_count;
        dd.write_offsets = dd.tasks2_offset;
        dd.write_vertices = dd.tasks2_vertices;
    }
    else {
        dd.write_count = dd.tasks1_count;
        dd.write_offsets = dd.tasks1_offset;
        dd.write_vertices = dd.tasks1_vertices;
    }

    // point of this is to find how many vertices will be transfered to tasks, it is easy to know how many tasks as it will just
    // be the expansion threshold, but to find how many vertices we must now the total size of all the tasks that will be copied.
    // each block does this but really could be done by one thread outside the GPU
    if (threadIdx.x == 0) {
        toffsetwrite = 0;
        twrite = 0;

        for (int i = 0; i < NUMBER_OF_WARPS; i++) {
            // if next warps count is more than expand threshold mark as such and break
            if (toffsetwrite + dd.wtasks_count[i] >= EXPAND_THRESHOLD) {
                twarp = i;
                break;
            }
            // else adds its size and count
            twrite += dd.wtasks_offset[(WTASKS_OFFSET_SIZE * i) + dd.wtasks_count[i]];
            toffsetwrite += dd.wtasks_count[i];
        }
        // final size is the size of all tasks up until last warp and the remaining tasks in the last warp until expand threshold is satisfied
        tasks_end = twrite + dd.wtasks_offset[(WTASKS_OFFSET_SIZE * twarp) + (EXPAND_THRESHOLD - toffsetwrite)];
    }
    __syncthreads();

    // warp level
    if (LANE_IDX == 0)
    {
        tasks_write[WIB_IDX] = 0;
        tasks_offset_write[WIB_IDX] = 1;
        cliques_write[WIB_IDX] = 0;
        cliques_offset_write[WIB_IDX] = 1;

        for (int i = 0; i < WARP_IDX; i++) {
            tasks_offset_write[WIB_IDX] += dd.wtasks_count[i];
            tasks_write[WIB_IDX] += dd.wtasks_offset[(WTASKS_OFFSET_SIZE * i) + dd.wtasks_count[i]];

            cliques_offset_write[WIB_IDX] += dd.wcliques_count[i];
            cliques_write[WIB_IDX] += dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * i) + dd.wcliques_count[i]];
        }
    }
    __syncwarp();
    
    // move to tasks and buffer
    for (int i = LANE_IDX + 1; i <= dd.wtasks_count[WARP_IDX]; i += WARP_SIZE)
    {
        if (tasks_offset_write[WIB_IDX] + i - 1 <= EXPAND_THRESHOLD) {
            // to tasks
            dd.write_offsets[tasks_offset_write[WIB_IDX] + i - 1] = dd.wtasks_offset[(WTASKS_OFFSET_SIZE * WARP_IDX) + i] + tasks_write[WIB_IDX];
        }
        else {
            // to buffer
            dd.buffer_offset[tasks_offset_write[WIB_IDX] + i - 2 - EXPAND_THRESHOLD + (*(dd.buffer_offset_start))] = dd.wtasks_offset[(WTASKS_OFFSET_SIZE * WARP_IDX) + i] +
                tasks_write[WIB_IDX] - tasks_end + (*(dd.buffer_start));
        }
    }

    for (int i = LANE_IDX; i < dd.wtasks_offset[(WTASKS_OFFSET_SIZE * WARP_IDX) + dd.wtasks_count[WARP_IDX]]; i += WARP_SIZE) {
        if (tasks_write[WIB_IDX] + i < tasks_end) {
            // to tasks
            dd.write_vertices[tasks_write[WIB_IDX] + i] = dd.wtasks_vertices[(WTASKS_SIZE * WARP_IDX) + i];
        }
        else {
            // to buffer
            dd.buffer_vertices[(*(dd.buffer_start)) + tasks_write[WIB_IDX] + i - tasks_end] = dd.wtasks_vertices[(WTASKS_SIZE * WARP_IDX) + i];
        }
    }

    //move to cliques
    for (int i = LANE_IDX + 1; i <= dd.wcliques_count[WARP_IDX]; i += WARP_SIZE) {
        dd.cliques_offset[(*(dd.cliques_offset_start)) + cliques_offset_write[WIB_IDX] + i - 2] = dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * WARP_IDX) + i] + (*(dd.cliques_start)) + 
            cliques_write[WIB_IDX];
    }
    for (int i = LANE_IDX; i < dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * WARP_IDX) + dd.wcliques_count[WARP_IDX]]; i += WARP_SIZE) {
        dd.cliques_vertex[(*(dd.cliques_start)) + cliques_write[WIB_IDX] + i] = dd.wcliques_vertex[(WCLIQUES_SIZE * WARP_IDX) + i];
    }

    if (IDX == 0) {
        // handle tasks and buffer counts
        if ((*dd.total_tasks) <= EXPAND_THRESHOLD) {
            (*dd.write_count) = (*(dd.total_tasks));
        }
        else {
            (*dd.write_count) = EXPAND_THRESHOLD;
            (*(dd.buffer_count)) += ((*(dd.total_tasks)) - EXPAND_THRESHOLD);
        }
        (*(dd.cliques_count)) += (*(dd.total_cliques));

        (*(dd.total_tasks)) = 0;
        (*(dd.total_cliques)) = 0;
    }
}

__global__ void fill_from_buffer(GPU_Data dd)
{
    if ((*(dd.current_level)) % 2 == 0) {
        dd.write_count = dd.tasks2_count;
        dd.write_offsets = dd.tasks2_offset;
        dd.write_vertices = dd.tasks2_vertices;
    }
    else {
        dd.write_count = dd.tasks1_count;
        dd.write_offsets = dd.tasks1_offset;
        dd.write_vertices = dd.tasks1_vertices;
    }

    // get read and write locations
    int write_amount = ((*(dd.buffer_count)) >= (EXPAND_THRESHOLD - (*dd.write_count))) ? EXPAND_THRESHOLD - (*dd.write_count) : (*(dd.buffer_count));
    uint64_t start_buffer = dd.buffer_offset[(*(dd.buffer_count)) - write_amount];
    uint64_t end_buffer = dd.buffer_offset[(*(dd.buffer_count))];
    uint64_t size_buffer = end_buffer - start_buffer;
    uint64_t start_write = dd.write_offsets[(*dd.write_count)];

    // handle offsets
    for (int i = IDX + 1; i <= write_amount; i += NUMBER_OF_THREADS) {
        dd.write_offsets[(*dd.write_count) + i] = start_write + (dd.buffer_offset[(*(dd.buffer_count)) - write_amount + i] - start_buffer);
    }

    // handle data
    for (int i = IDX; i < size_buffer; i += NUMBER_OF_THREADS) {
        dd.write_vertices[start_write + i] = dd.buffer_vertices[start_buffer + i];
    }

    if (IDX == 0) {
        (*dd.write_count) += write_amount;
        (*(dd.buffer_count)) -= write_amount;
    }
}