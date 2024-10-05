#include "../inc/common.hpp"
#include "../inc/host_functions.hpp"
#include "../inc/host_debug.h"
#include "../inc/device_kernels.hpp"
#include "../inc/cuTS_MPI.h"

// --- PRIMARY FUNCTIONS ---
// initializes minimum degrees array 
void h_calculate_minimum_degrees(CPU_Graph& hg, int* minimum_degrees, double minimum_degree_ratio)
{
    minimum_degrees[0] = 0;
    for (int i = 1; i <= hg.number_of_vertices; i++) {
        minimum_degrees[i] = ceil(minimum_degree_ratio * (i - 1));
    }
}

void h_search(CPU_Graph& hg, ofstream& temp_results, DS_Sizes& dss, int* minimum_out_degrees, 
               int* minimum_in_degrees, double minimum_out_degree_ratio, 
               double minimum_in_degree_ratio, int minimum_clique_size, string output) 
{
    CPU_Data hd;                    // host vertex structure data
    CPU_Cliques hc;                 // host results data
    GPU_Data h_dd;                  // host pointers to device global memory
    GPU_Data* dd;                   // device pointers to device global memory
    uint64_t* mpiSizeBuffer;        // where data transfered intranode is stored
    Vertex* mpiVertexBuffer;        // vertex intranode data
    bool help_others;               // whether node helped other
    int taker;                      // taker node id for debugging
    bool divided_work;              // whether node gave work
    int from;                       // sending node id for debugging
    uint64_t* tasks_count;          // unified memory for tasks count
    uint64_t* buffer_count;         // unified memory for buffer count
    uint64_t* cliques_count;        // unified memory for cliques count
    uint64_t* cliques_size;
    uint64_t* write_count;

    // TIME
    auto start = chrono::high_resolution_clock::now();
    if(grank == 0){
        cout << "INITIALIZING TASKS:  " << flush;
    }

    // MPI
    mpiSizeBuffer = new uint64_t[MAX_MESSAGE];
    mpiVertexBuffer = new Vertex[MAX_MESSAGE];
    // open communication channels
    mpi_irecv_all(grank);
    for (int i = 0; i < wsize; ++i) {
        global_free_list[i] = false;
    }

    // HANDLE MEMORY
    h_allocate_host_memory(hd, h_dd, hc, hg, dss, minimum_out_degrees, minimum_in_degrees, 
                       minimum_out_degree_ratio, minimum_in_degree_ratio, minimum_clique_size);

    // INITIALIZE TASKS
    h_initialize_tasks(hg, hd, minimum_out_degrees, minimum_in_degrees, minimum_clique_size);

    // DEBUG
    if (dss.DEBUG_TOGGLE) {
        output_file << "CONDENSED GRAPH" << endl;
        print_graph(hg);
    }

    // HANDLE MEMORY
    h_allocate_device_memory(hd, h_dd, hg, dss, minimum_out_degrees, minimum_in_degrees, 
                             minimum_out_degree_ratio, minimum_in_degree_ratio, 
                             minimum_clique_size);
    chkerr(cudaMalloc((void**)&dd, sizeof(GPU_Data)));
    chkerr(cudaMemcpy(dd, &h_dd, sizeof(GPU_Data), cudaMemcpyHostToDevice));
    chkerr(cudaMallocManaged((void**)&tasks_count, sizeof(uint64_t)));
    chkerr(cudaMallocManaged((void**)&buffer_count, sizeof(uint64_t)));
    chkerr(cudaMallocManaged((void**)&cliques_count, sizeof(uint64_t)));
    chkerr(cudaMallocManaged((void**)&cliques_size, sizeof(uint64_t)));

    // DEBUG
    if (dss.DEBUG_TOGGLE) {
        mvs = (*(hd.tasks1_offset + (*hd.tasks1_count)));
        if (mvs > dss.WVERTICES_SIZE) {
            cout << "!!! VERTICES SIZE ERROR !!!" << endl;
            return;
        }
        output_file << "CPU START" << endl;
        print_H_Data_Sizes(hd, hc);
    }

    // CPU EXPANSION
    // cpu expand must be called atleast one time to handle first round cover pruning as the gpu 
    // code cannot do this
    for (int i = 0; i < CPU_LEVELS + 1 && !(*hd.maximal_expansion); i++) {
        
        *hd.maximal_expansion = true;

        // EXPAND LEVEL
        // will set maximal expansion false if no work generated
        h_expand_level(hg, hd, hc, dss, minimum_out_degrees, minimum_in_degrees, 
                       minimum_out_degree_ratio, minimum_in_degree_ratio, minimum_clique_size);

        // FILL TASKS FROM BUFFER
        // determine the write location for this level
        if ((*hd.current_level) % 2 == 0) {
            write_count = hd.tasks2_count;
        }
        else {
            write_count = hd.tasks1_count;
        }

        // if last CPU round copy enough tasks for GPU expansion threshold
        if ((*hd.current_level) == CPU_LEVELS && CPU_EXPAND_THRESHOLD < dss.EXPAND_THRESHOLD && 
            (*hd.buffer_count) > 0) {

            h_fill_from_buffer(hd, dss.EXPAND_THRESHOLD);
        }
        // if not enough generated to fully populate fill from buffer to cpu expand threshold
        else if (*write_count < CPU_EXPAND_THRESHOLD && (*hd.buffer_count) > 0){
            h_fill_from_buffer(hd, CPU_EXPAND_THRESHOLD);
        }

        // FINISH LEVEL
        // determine whether maximal expansion has been accomplished, variables changed in kernel
        if (*write_count > 0) {
            *hd.maximal_expansion = false;
        }

        (*hd.current_level)++;
    
        // if cliques is more than threshold dump
        if (*hc.cliques_count > (int)(dss.CLIQUES_OFFSET_SIZE * (dss.CLIQUES_PERCENT / 100.0)) || 
                hc.cliques_offset[*hc.cliques_count] > (int)(dss.CLIQUES_SIZE * 
                (dss.CLIQUES_PERCENT / 100.0))) {

            h_flush_cliques(hc, temp_results);
        }

        // DEBUG
        if (dss.DEBUG_TOGGLE) {
            print_H_Data_Sizes(hd, hc);
        }
    }

    h_flush_cliques(hc, temp_results);

    // TIME
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    if(grank == 0){
        cout << duration.count() << " ms" << endl;
    }
    start = chrono::high_resolution_clock::now();

    // only bother doing gpu initialization steps if program wasnt completed on CPU
    if(!*hd.maximal_expansion){

        // TRANSFER TO GPU
        h_move_to_gpu(hd, h_dd, dss, output);
    }

    // DEBUG
    if(grank == 0){
        cout << "EXPANDING TASKS:     ";
    }

    help_others = false;

    // wait after all work in process has been completed, loop if work has been given from another 
    // process, break if all process complete work
    do{

        // HELP OTHER PROCESS
        // only way this variable is true is if this process got finished all of its work broke out
        // of inner loop and returned here
        if(help_others){
            // decode buffer
            decode_com_buffer(h_dd, mpiSizeBuffer, mpiVertexBuffer);
            // populate tasks from buffer
            d_fill_from_buffer<<<NUMBER_OF_BLOCKS, BLOCK_SIZE>>>(dd, tasks_count, buffer_count);
            cudaDeviceSynchronize();
            *hd.maximal_expansion = false;

            // DEBUG
            if (dss.DEBUG_TOGGLE) {
                output_file << "RECIEVING WORK FROM PROCESS " << from << endl;
                print_D_Data_Sizes(h_dd, dss);
            }
        }

        // HANDLE LOCAL WORK
        // loop while not all work has been completed
        while (!*hd.maximal_expansion){
            
            *hd.maximal_expansion = true;

            // EXPAND LEVEL
            // expand all tasks in 'tasks' array, each warp will write to their respective warp 
            // tasks buffer in global memory
            d_expand_level<<<NUMBER_OF_BLOCKS, BLOCK_SIZE>>>(dd);
            cudaDeviceSynchronize();

            // DEBUG
            if (dss.DEBUG_TOGGLE) {
                print_D_Warp_Data_Sizes(h_dd, dss);
            }

            // consolidate all the warp tasks/cliques buffers into the next global tasks array, 
            // buffer, and cliques
            d_transfer_buffers<<<NUMBER_OF_BLOCKS, BLOCK_SIZE>>>(dd, tasks_count, buffer_count, 
                                                                 cliques_count, cliques_size);
            cudaDeviceSynchronize();

            // FILL TASKS FROM BUFFER
            if (*tasks_count < dss.EXPAND_THRESHOLD && *buffer_count > 0) {
                // if not enough tasks were generated when expanding the previous level to fill the 
                // next tasks array the program will attempt to fill the tasks array by popping 
                // tasks from the buffer
                d_fill_from_buffer<<<NUMBER_OF_BLOCKS, BLOCK_SIZE>>>(dd, tasks_count, buffer_count);
                cudaDeviceSynchronize();
            }

            // FINISH LEVEL
            // determine whether maximal expansion has been accomplished, variables changed in 
            // kernel
            if (*tasks_count > 0) {
                *hd.maximal_expansion = false;
            }

            // determine whether cliques has exceeded defined threshold, if so dump them to a file, 
            // variables changed in kernel
            if (*cliques_count > (int)(dss.CLIQUES_OFFSET_SIZE * (dss.CLIQUES_PERCENT / 100.0)) || 
                *cliques_size > (int)(dss.CLIQUES_SIZE * (dss.CLIQUES_PERCENT / 100.0))) {

                h_dump_cliques(hc, h_dd, temp_results, dss);
            }

            // DEBUG
            if (dss.DEBUG_TOGGLE) {
                print_D_Data_Sizes(h_dd, dss);
            }

            // // GET HELP FROM OTHER PROCESS
            // if(*buffer_count > HELP_THRESHOLD){

            //     // DEBUG - rm
            //     cout << "1" << endl;

            //     // return whether work was successfully given
            //     divided_work = give_work_wrapper(grank, taker, mpiSizeBuffer, mpiVertexBuffer, 
            //                                      h_dd, *buffer_count, dss);

            //     // DEBUG - rm
            //     cout << "2" << endl;

            //     // update buffer count if work was given
            //     if(divided_work){
            //         *buffer_count -= (*buffer_count > dss.EXPAND_THRESHOLD) ? dss.EXPAND_THRESHOLD 
            //         + ((*buffer_count - dss.EXPAND_THRESHOLD) * ((100 - HELP_PERCENT) / 100.0)) : 
            //         *buffer_count;

            //         chkerr(cudaMemcpy(h_dd.buffer_count, buffer_count, sizeof(uint64_t), 
            //                           cudaMemcpyHostToDevice));

            //         // DEBUG
            //         if (dss.DEBUG_TOGGLE) {
            //             output_file << "SENDING WORK TO PROCESS " << taker << endl;
            //             print_D_Data_Sizes(h_dd, dss);
            //         }
            //     }
            // }
        }

        // we have finished all our work, so if we get to the top of the loop again it is because 
        // we are helping someone else
        help_others = true;

    // INITIATE HELP FOR OTHER PROCESS
    // each process will block here until they have found another process to recieve work from and
    // then help or all processes are done and the program can complete
    }while(wsize != take_work_wrap(grank, mpiSizeBuffer, mpiVertexBuffer, from));

    h_dump_cliques(hc, h_dd, temp_results, dss);

    // clean up
    h_free_memory(hd, h_dd, hc);
    chkerr(cudaFree(dd));
    chkerr(cudaFree(tasks_count));
    chkerr(cudaFree(buffer_count));
    chkerr(cudaFree(cliques_count));

    // TIME
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    if(grank == 0){
        cout << duration.count() << " ms" << endl;
    }
}

//allocates memory for the data structures on the host and device   
void h_allocate_host_memory(CPU_Data& hd, GPU_Data& h_dd, CPU_Cliques& hc, CPU_Graph& hg, 
                        DS_Sizes& dss, int* minimum_out_degrees, int* minimum_in_degrees, 
                        double minimum_out_degree_ratio, double minimum_in_degree_ratio, 
                        int minimum_clique_size)
{
    // CPU DATA
    hd.tasks1_count = new uint64_t;
    hd.tasks1_offset = new uint64_t[dss.EXPAND_THRESHOLD + 1];
    hd.tasks1_vertices = new Vertex[dss.TASKS_SIZE];
    hd.tasks1_offset[0] = 0;
    (*(hd.tasks1_count)) = 0;
    hd.tasks2_count = new uint64_t;
    hd.tasks2_offset = new uint64_t[dss.EXPAND_THRESHOLD + 1];
    hd.tasks2_vertices = new Vertex[dss.TASKS_SIZE];
    hd.tasks2_offset[0] = 0;
    (*(hd.tasks2_count)) = 0;
    hd.buffer_count = new uint64_t;
    hd.buffer_offset = new uint64_t[dss.BUFFER_OFFSET_SIZE];
    hd.buffer_vertices = new Vertex[dss.BUFFER_SIZE];
    hd.buffer_offset[0] = 0;
    (*(hd.buffer_count)) = 0;
    hd.current_level = new uint64_t;
    hd.maximal_expansion = new bool;
    (*hd.current_level) = 0;
    (*hd.maximal_expansion) = false;
    hd.vertex_order_map = new int[hg.number_of_vertices];
    hd.remaining_candidates = new int[hg.number_of_vertices];
    hd.removed_candidates = new int[hg.number_of_vertices];
    hd.remaining_count = new int;
    hd.removed_count = new int;
    hd.candidate_out_mem_degs = new int[hg.number_of_vertices];
    hd.candidate_in_mem_degs = new int[hg.number_of_vertices];
    memset(hd.vertex_order_map, -1, sizeof(int) * hg.number_of_vertices);
    // CPU CLIQUES
    hc.cliques_count = new uint64_t;
    hc.cliques_vertex = new int[dss.CLIQUES_SIZE];
    hc.cliques_offset = new uint64_t[dss.CLIQUES_OFFSET_SIZE];
    hc.cliques_offset[0] = 0;
    (*(hc.cliques_count)) = 0;
}

void h_allocate_device_memory(CPU_Data& hd, GPU_Data& h_dd, CPU_Graph& hg, DS_Sizes& dss, 
                              int* minimum_out_degrees, int* minimum_in_degrees, 
                              double minimum_out_degree_ratio, double minimum_in_degree_ratio, 
                              int minimum_clique_size)
{
    // GPU GRAPH
    chkerr(cudaMalloc((void**)&h_dd.number_of_vertices, sizeof(int)));
    chkerr(cudaMalloc((void**)&h_dd.number_of_edges, sizeof(int)));
    chkerr(cudaMalloc((void**)&h_dd.out_neighbors, sizeof(int) * hg.number_of_edges));
    chkerr(cudaMalloc((void**)&h_dd.out_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&h_dd.in_neighbors, sizeof(int) * hg.number_of_edges));
    chkerr(cudaMalloc((void**)&h_dd.in_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&h_dd.twohop_neighbors, sizeof(int) * hg.number_of_lvl2adj));
    chkerr(cudaMalloc((void**)&h_dd.twohop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1)));
    chkerr(cudaMemcpy(h_dd.number_of_vertices, &(hg.number_of_vertices), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.number_of_edges, &(hg.number_of_edges), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.out_neighbors, hg.out_neighbors, sizeof(int) * hg.number_of_edges, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.out_offsets, hg.out_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.in_neighbors, hg.in_neighbors, sizeof(int) * hg.number_of_edges, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.in_offsets, hg.in_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.twohop_neighbors, hg.twohop_neighbors, sizeof(int) * hg.number_of_lvl2adj, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.twohop_offsets, hg.twohop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
    // GPU DATA
    chkerr(cudaMalloc((void**)&h_dd.current_level, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.tasks_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.tasks_offset, sizeof(uint64_t) * (dss.EXPAND_THRESHOLD + 1)));
    chkerr(cudaMalloc((void**)&h_dd.tasks_vertices, sizeof(Vertex) * dss.TASKS_SIZE));
    chkerr(cudaMemset(h_dd.tasks_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(h_dd.tasks_count, 0, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.buffer_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.buffer_offset, sizeof(uint64_t) * dss.BUFFER_OFFSET_SIZE));
    chkerr(cudaMalloc((void**)&h_dd.buffer_vertices, sizeof(Vertex) * dss.BUFFER_SIZE));
    chkerr(cudaMemset(h_dd.buffer_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(h_dd.buffer_count, 0, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.wtasks_count, sizeof(uint64_t) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.wtasks_offset, (sizeof(uint64_t) * dss.WTASKS_OFFSET_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.wtasks_vertices, (sizeof(Vertex) * dss.WTASKS_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMemset(h_dd.wtasks_offset, 0, (sizeof(uint64_t) * dss.WTASKS_OFFSET_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.global_vertices, (sizeof(Vertex) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.removed_candidates, (sizeof(int) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.lane_removed_candidates, (sizeof(int) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.remaining_candidates, (sizeof(int) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.temp_vertex_array, (sizeof(Vertex) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.lane_remaining_candidates, (sizeof(int) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.candidate_out_mem_degs, (sizeof(int) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.lane_candidate_out_mem_degs, (sizeof(int) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.candidate_in_mem_degs, (sizeof(int) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.lane_candidate_in_mem_degs, (sizeof(int) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.adjacencies, (sizeof(int) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.minimum_out_degree_ratio, sizeof(double)));
    chkerr(cudaMalloc((void**)&h_dd.minimum_out_degrees, sizeof(int) * (hg.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&h_dd.minimum_in_degree_ratio, sizeof(double)));
    chkerr(cudaMalloc((void**)&h_dd.minimum_in_degrees, sizeof(int) * (hg.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&h_dd.minimum_clique_size, sizeof(int)));
    chkerr(cudaMemcpy(h_dd.minimum_out_degree_ratio, &minimum_out_degree_ratio, sizeof(double), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.minimum_out_degrees, minimum_out_degrees, sizeof(int) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.minimum_in_degree_ratio, &minimum_in_degree_ratio, sizeof(double), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.minimum_in_degrees, minimum_in_degrees, sizeof(int) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.minimum_clique_size, &minimum_clique_size, sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMalloc((void**)&h_dd.total_tasks, sizeof(int)));
    chkerr(cudaMemset(h_dd.total_tasks, 0, sizeof(int)));
    chkerr(cudaMalloc((void**)&h_dd.vertex_order_map, (sizeof(int) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS));
    int* vertex_order_map = new int[dss.WVERTICES_SIZE * NUMBER_OF_WARPS];
    memset(vertex_order_map, -1, sizeof(int) * dss.WVERTICES_SIZE * NUMBER_OF_WARPS);
    chkerr(cudaMemcpy(h_dd.vertex_order_map, vertex_order_map, (sizeof(int) * dss.WVERTICES_SIZE) * NUMBER_OF_WARPS, cudaMemcpyHostToDevice));
    // GPU CLIQUES
    chkerr(cudaMalloc((void**)&h_dd.cliques_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.cliques_vertex, sizeof(int) * dss.CLIQUES_SIZE));
    chkerr(cudaMalloc((void**)&h_dd.cliques_offset, sizeof(uint64_t) * dss.CLIQUES_OFFSET_SIZE));
    chkerr(cudaMemset(h_dd.cliques_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(h_dd.cliques_count, 0, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.wcliques_count, sizeof(uint64_t) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.wcliques_offset, (sizeof(uint64_t) * dss.WCLIQUES_OFFSET_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.wcliques_vertex, (sizeof(int) * dss.WCLIQUES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMemset(h_dd.wcliques_offset, 0, (sizeof(uint64_t) * dss.WCLIQUES_OFFSET_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.total_cliques, sizeof(int)));
    chkerr(cudaMemset(h_dd.total_cliques, 0, sizeof(int)));
    chkerr(cudaMalloc((void**)&h_dd.buffer_offset_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.buffer_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.cliques_offset_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.cliques_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.current_task, sizeof(int)));
    int current = NUMBER_OF_WARPS;
    int* pcurrent = &current;
    chkerr(cudaMemcpy(h_dd.current_task, pcurrent, sizeof(int), cudaMemcpyHostToDevice));
    // DATA STRUCTURE SIZES
    chkerr(cudaMalloc((void**)&h_dd.TASKS_SIZE, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.TASKS_PER_WARP, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.BUFFER_SIZE, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.BUFFER_OFFSET_SIZE, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.CLIQUES_SIZE, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.CLIQUES_OFFSET_SIZE, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.CLIQUES_PERCENT, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.WCLIQUES_SIZE, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.WCLIQUES_OFFSET_SIZE, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.WTASKS_SIZE, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.WTASKS_OFFSET_SIZE, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.WVERTICES_SIZE, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.EXPAND_THRESHOLD, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.CLIQUES_DUMP, sizeof(uint64_t)));
    chkerr(cudaMemcpy(h_dd.TASKS_SIZE, &dss.TASKS_SIZE, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.TASKS_PER_WARP, &dss.TASKS_PER_WARP, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.BUFFER_SIZE, &dss.BUFFER_SIZE, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.BUFFER_OFFSET_SIZE, &dss.BUFFER_OFFSET_SIZE, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.CLIQUES_SIZE, &dss.CLIQUES_SIZE, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.CLIQUES_OFFSET_SIZE, &dss.CLIQUES_OFFSET_SIZE, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.CLIQUES_PERCENT, &dss.CLIQUES_PERCENT, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.WCLIQUES_SIZE, &dss.WCLIQUES_SIZE, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.WCLIQUES_OFFSET_SIZE, &dss.WCLIQUES_OFFSET_SIZE, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.WTASKS_SIZE, &dss.WTASKS_SIZE, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.WTASKS_OFFSET_SIZE, &dss.WTASKS_OFFSET_SIZE, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.WVERTICES_SIZE, &dss.WVERTICES_SIZE, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.EXPAND_THRESHOLD, &dss.EXPAND_THRESHOLD, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.CLIQUES_DUMP, &dss.CLIQUES_DUMP, sizeof(uint64_t), cudaMemcpyHostToDevice));
}

// processes 0th level of expansion
void h_initialize_tasks(CPU_Graph& hg, CPU_Data& hd, int* minimum_out_degrees, 
                        int* minimum_in_degrees, int minimum_clique_size)
{
    // intersection
    int pvertexid;                      // vertex id for pruning
    uint64_t pneighbors_start;          // start of neighbors for pruning
    uint64_t pneighbors_end;            // end of neighbors for pruning
    int phelper1;                       // helper for pruning
    int maximum_degree;                 // maximum degree of any vertex
    int maximum_degree_index;           // index in vertices of max vertex
    int total_vertices;                 // total vertices
    int number_of_candidates;           // number of candidate vertices
    Vertex* vertices;                   // array of vertices
    int removed_start;                  // tracks how many updates done for pruning
    int min_out_deg;
    int min_in_deg;
    int vertex_deg;
    int* temp_array;

    temp_array = new int[hg.number_of_vertices];
    memset(temp_array, 0, sizeof(int) * hg.number_of_vertices);

    min_out_deg = minimum_out_degrees[minimum_clique_size];
    min_in_deg = minimum_in_degrees[minimum_clique_size];

    (*hd.remaining_count) = 0;
    (*hd.removed_count) = 0;

    // INITIALIZE VERTICES
    total_vertices = hg.number_of_vertices;
    vertices = new Vertex[total_vertices];
    number_of_candidates = total_vertices;

    // set all vertex information
    for (int i = 0; i < total_vertices; i++) {
        vertices[i].vertexid = i;
        vertices[i].out_mem_deg = 0;
        vertices[i].out_can_deg = hg.out_offsets[i + 1] - hg.out_offsets[i];
        vertices[i].in_mem_deg = 0;
        vertices[i].in_can_deg = hg.in_offsets[i + 1] - hg.in_offsets[i];
        vertices[i].lvl2adj = hg.twohop_offsets[i + 1] - hg.twohop_offsets[i];

        // see whether vertex is valid: -1 is pruned, 0 is candidate
        if (vertices[i].out_can_deg >= min_out_deg && vertices[i].in_can_deg >= min_in_deg && 
            vertices[i].lvl2adj >= minimum_clique_size - 1) {

            vertices[i].label = 0;
            hd.remaining_candidates[(*hd.remaining_count)++] = i;
        }
        else {
            vertices[i].label = -1;
            hd.removed_candidates[(*hd.removed_count)++] = i;
        }
    }

    // DEGREE-BASED PRUNING
    // update while half of vertices have been removed
    while ((*hd.remaining_count) < number_of_candidates / 2) {
        number_of_candidates = (*hd.remaining_count);
        
        for (int i = 0; i < number_of_candidates; i++) {
            vertices[hd.remaining_candidates[i]].out_can_deg = 0;
            vertices[hd.remaining_candidates[i]].in_can_deg = 0;
        }

        // iterate through all remaining vertices and use them to update degrees
        for (int i = 0; i < number_of_candidates; i++) {
            // in 0th level id is same as position in vertices as all vertices are in vertices, see
            // last block
            pvertexid = hd.remaining_candidates[i];

            // update using out degrees
            pneighbors_start = hg.out_offsets[pvertexid];
            pneighbors_end = hg.out_offsets[pvertexid + 1];

            for (int j = pneighbors_start; j < pneighbors_end; j++) {

                phelper1 = hg.out_neighbors[j];

                if (vertices[phelper1].label == 0) {
                    vertices[phelper1].in_can_deg++;
                }
            }

            // update using in degrees
            pneighbors_start = hg.in_offsets[pvertexid];
            pneighbors_end = hg.in_offsets[pvertexid + 1];

            for (int j = pneighbors_start; j < pneighbors_end; j++) {
                
                phelper1 = hg.in_neighbors[j];

                if (vertices[phelper1].label == 0) {
                    vertices[phelper1].out_can_deg++;
                }
            }
        }

        (*hd.remaining_count) = 0;
        (*hd.removed_count) = 0;

        // remove more vertices based on updated degrees
        for (int i = 0; i < number_of_candidates; i++) {
            phelper1 = hd.remaining_candidates[i];

            if (vertices[phelper1].out_can_deg >= min_out_deg && vertices[phelper1].in_can_deg >= 
                min_in_deg) {

                hd.remaining_candidates[(*hd.remaining_count)++] = phelper1;
            }
            else {
                vertices[phelper1].label = -1;
                hd.removed_candidates[(*hd.removed_count)++] = phelper1;
            }
        }
    }

    number_of_candidates = (*hd.remaining_count);

    // update degrees based on last round of removed vertices
    removed_start = 0;
    while((*hd.removed_count) > removed_start) {
        pvertexid = hd.removed_candidates[removed_start];

        pneighbors_start = hg.out_offsets[pvertexid];
        pneighbors_end = hg.out_offsets[pvertexid + 1];

        for (int j = pneighbors_start; j < pneighbors_end; j++) {

            phelper1 = hg.out_neighbors[j];

            if (vertices[phelper1].label == 0) {
                vertices[phelper1].in_can_deg--;

                if (vertices[phelper1].in_can_deg < min_in_deg) {
                    vertices[phelper1].label = -1;
                    number_of_candidates--;
                    hd.removed_candidates[(*hd.removed_count)++] = phelper1;
                }
            }
        }

        pneighbors_start = hg.in_offsets[pvertexid];
        pneighbors_end = hg.in_offsets[pvertexid + 1];

        for (int j = pneighbors_start; j < pneighbors_end; j++) {

            phelper1 = hg.in_neighbors[j];

            if (vertices[phelper1].label == 0) {
                vertices[phelper1].out_can_deg--;

                if (vertices[phelper1].out_can_deg < min_out_deg) {
                    vertices[phelper1].label = -1;
                    number_of_candidates--;
                    hd.removed_candidates[(*hd.removed_count)++] = phelper1;
                }
            }
        }

        removed_start++;
    }
    
    // FIRST ROUND COVER PRUNING
    // find cover vertex
    maximum_degree = 0;
    maximum_degree_index = 0;

    for (int i = 0; i < total_vertices; i++) {
        if (vertices[i].label == 0) {

            vertex_deg = min(vertices[i].out_can_deg, vertices[i].in_can_deg);

            if (vertex_deg > maximum_degree) {
                maximum_degree = vertex_deg;
                maximum_degree_index = i;
            }
        }
    }
    vertices[maximum_degree_index].label = 3;

    // find all covered vertices
    pneighbors_start = hg.out_offsets[maximum_degree_index];
    pneighbors_end = hg.out_offsets[maximum_degree_index + 1];

    for (int i = pneighbors_start; i < pneighbors_end; i++) {

        pvertexid = hg.out_neighbors[i];

        if (vertices[pvertexid].label == 0) {
            temp_array[pvertexid] = 1;
        }
    }

    pneighbors_start = hg.in_offsets[maximum_degree_index];
    pneighbors_end = hg.in_offsets[maximum_degree_index + 1];

    for (int i = pneighbors_start; i < pneighbors_end; i++) {

        pvertexid = hg.in_neighbors[i];

        if (temp_array[pvertexid] == 1) {
            vertices[pvertexid].label = 2;
        }
    }

    // sort enumeration order before writing to tasks
    qsort(vertices, total_vertices, sizeof(Vertex), h_comp_vert_Q);

    // DEBUG - uncomment
    //h_condense_graph(hd, hg, vertices, number_of_candidates);

    total_vertices = number_of_candidates;

    // WRITE TO TASKS
    if (total_vertices > 0)
    {
        for (int j = 0; j < total_vertices; j++) {
            hd.tasks1_vertices[j] = vertices[j];
        }
        (*(hd.tasks1_count))++;
        hd.tasks1_offset[(*(hd.tasks1_count))] = total_vertices;
    }

    delete[] vertices;
    delete[] temp_array;
}

// after initialization we can condense graph so we do not have to consider vertices that were
// pruned in 0th level and thus will not be needed by any task
void h_condense_graph(CPU_Data& hd, CPU_Graph& hg, Vertex* vertices, int number_of_candidates)
{
    uint64_t pneighbors_start;          // start of neighbors for pruning
    uint64_t pneighbors_end;
    int i, j, nvertex_no, norder;
    int* index2id;
    int number_of_edges = 0;
    int number_of_lvl2adj = 0;
	
	// initialize vertex map
	for(i=0;i<number_of_candidates;i++)
		hd.vertex_order_map[vertices[i].vertexid] = i;

	// declare and initialize new adj arrays
	int nlist_len, ncand_nbs, **ppnew_adjlists_o, **ppnew_adjlists_i, **ppnew_lvl2_nbs, v_index;
	ppnew_adjlists_o = new int*[number_of_candidates];
	ppnew_adjlists_i = new int*[number_of_candidates];
	ppnew_lvl2_nbs = new int*[number_of_candidates];

    int* gptemp_array = new int[hg.number_of_vertices];

	//translate candidate vertex id to index

	// id2index[i] = j, means vertex with id i is in position k of vertices
	map<int, int> id2index_map;

	// index2id[i] = j, means that vertex at position i of vertices has id k
	index2id = new int[number_of_candidates];

	// for all vertices
	for(i=0;i<number_of_candidates;i++)
	{
		// get vertex id
		nvertex_no = vertices[i].vertexid;
		
		// set index2id position i (index in vertices array) to the id of the given vertex
		index2id[i] = nvertex_no;

		// in id2index map the id of the given vertex to position i (idex in vertices array)
		id2index_map[nvertex_no] = i;
	}

	//reset 2hop adj
	// for all vertices
	for(i=0;i<number_of_candidates;i++)
	{
		nlist_len = 0;
		ncand_nbs = 0;

		// get vertex id
		nvertex_no = vertices[i].vertexid;

		// get vertex index in vertices
		v_index = id2index_map[nvertex_no];
        pneighbors_start = hg.twohop_offsets[nvertex_no];
        pneighbors_end = hg.twohop_offsets[nvertex_no + 1];

        // for all twohop adj
        for(j=pneighbors_start;j<pneighbors_end;j++)
        {
            // get index in vertices of twohop adj
            norder = hd.vertex_order_map[hg.twohop_neighbors[j]];

            // if twohop adj was in vertices
            if(norder>=0)
            {
                //add nb's translated id
                // add the translated id of the twohop adj to the vertices new twohop adj list
                gptemp_array[nlist_len++] = id2index_map[hg.twohop_neighbors[j]];

                // not sure why this variable exists has same value as nlist_len
                ncand_nbs++;

                number_of_lvl2adj++;
            }
        }

		// if vertex had some twohop adj
		if(nlist_len>0)
		{
			// make new array for vertex
			ppnew_lvl2_nbs[v_index] = new int[nlist_len+1];

			// set array size indicator
			ppnew_lvl2_nbs[v_index][0] = nlist_len;

			// copy twohop adj to array
			memcpy(&ppnew_lvl2_nbs[v_index][1], gptemp_array, sizeof(int)*nlist_len);

			// sort array
			qsort(&ppnew_lvl2_nbs[v_index][1], ppnew_lvl2_nbs[i][0], sizeof(int), comp_int);
		}

		// as byproduct of condensing graph we calculate lvl2adj of vertices
        // used for lookahead pruning
		vertices[i].lvl2adj = ncand_nbs;
	}


	//reset 1hop adj
	// for all vertices
	for(i=0;i<number_of_candidates;i++)
	{
		// reset out-direction
		nlist_len = 0;
		ncand_nbs = 0;

		// get vertex id and index
		nvertex_no = vertices[i].vertexid;
		v_index = id2index_map[nvertex_no];

        pneighbors_start = hg.out_offsets[nvertex_no];
        pneighbors_end = hg.out_offsets[nvertex_no + 1];

        // for all out adj
        for(j=pneighbors_start;j<pneighbors_end;j++)
        {
            // get out adj position in vertices
            norder = hd.vertex_order_map[hg.out_neighbors[j]];

            // if it exists in vertices
            if(norder>=0)
            {
                // add out adj translated id to new out adj
                gptemp_array[nlist_len++] = id2index_map[hg.out_neighbors[j]];
                ncand_nbs++;

                number_of_edges++;
            }
        }

		//check cand degree
		if(vertices[i].out_can_deg!=ncand_nbs)
			printf("Error: inconsistent candidate degree\n");

		// if some new out adj found copy to new main list
		if(nlist_len>0)
		{
			ppnew_adjlists_o[v_index] = new int[nlist_len+1];
			ppnew_adjlists_o[v_index][0] = nlist_len;
			memcpy(&ppnew_adjlists_o[v_index][1], gptemp_array, sizeof(int)*nlist_len);
			qsort(&ppnew_adjlists_o[v_index][1], ppnew_adjlists_o[i][0], sizeof(int), comp_int);
		}

		// reset in-direction
		nlist_len = 0;
		ncand_nbs = 0;

        pneighbors_start = hg.in_offsets[nvertex_no];
        pneighbors_end = hg.in_offsets[nvertex_no + 1];

        // for all in adj
        for(j=pneighbors_start;j<pneighbors_end;j++)
        {
            // get position of in adj in vertices
            norder = hd.vertex_order_map[hg.in_neighbors[j]];

            // if in adj is in vertices
            if(norder>=0)
            {
                // add in aj translated id to new in adj
                gptemp_array[nlist_len++] = id2index_map[hg.in_neighbors[j]];
                ncand_nbs++;
            }
        }

		//check cand degree
		if(vertices[i].in_can_deg!=ncand_nbs)
			printf("Error: inconsistent candidate degree\n");

		// if therer were some in adj copy to new main list
		if(nlist_len>0)
		{
			ppnew_adjlists_i[v_index] = new int[nlist_len+1];
			ppnew_adjlists_i[v_index][0] = nlist_len;
			memcpy(&ppnew_adjlists_i[v_index][1], gptemp_array, sizeof(int)*nlist_len);
			qsort(&ppnew_adjlists_i[v_index][1], ppnew_adjlists_i[i][0], sizeof(int), comp_int);
		}

	}

    // reset vertex order map
    for (int i = 0; i < number_of_candidates; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = -1;
    }

	//translate pvertex
	// for all vertices
	for(i=0;i<number_of_candidates;i++)
		// convert vertex id to translated id
		vertices[i].vertexid = i;

	// transfer twohop offsets
	hg.twohop_offsets[0] = 0;
	for(int i = 0; i < number_of_candidates; i++){
		hg.twohop_offsets[i + 1] = hg.twohop_offsets[i] + ppnew_lvl2_nbs[i][0];
	}
	// transfer twohop neighbors
	for(int i = 0; i < number_of_candidates; i++){
		for(uint64_t k = 0; k < hg.twohop_offsets[i + 1] - hg.twohop_offsets[i]; k++){
			hg.twohop_neighbors[hg.twohop_offsets[i] + k] = ppnew_lvl2_nbs[i][k + 1];
		}
	}

    // transfer out offsets
	hg.out_offsets[0] = 0;
	for(int i = 0; i < number_of_candidates; i++){
		hg.out_offsets[i + 1] = hg.out_offsets[i] + ppnew_adjlists_o[i][0];
	}
	// transfer out neighbors
	for(int i = 0; i < number_of_candidates; i++){
		for(uint64_t k = 0; k < hg.out_offsets[i + 1] - hg.out_offsets[i]; k++){
			hg.out_neighbors[hg.out_offsets[i] + k] = ppnew_adjlists_o[i][k + 1];
		}
	}

    // transfer out offsets
	hg.in_offsets[0] = 0;
	for(int i = 0; i < number_of_candidates; i++){
		hg.in_offsets[i + 1] = hg.in_offsets[i] + ppnew_adjlists_i[i][0];
	}
	// transfer out neighbors
	for(int i = 0; i < number_of_candidates; i++){
		for(uint64_t k = 0; k < hg.in_offsets[i + 1] - hg.in_offsets[i]; k++){
			hg.in_neighbors[hg.in_offsets[i] + k] = ppnew_adjlists_i[i][k + 1];
		}
	}

    delete[] index2id;
    delete[] gptemp_array;
    for(i=0;i<number_of_candidates;i++){
        delete[] ppnew_lvl2_nbs[i];
        delete[] ppnew_adjlists_o[i];
        delete[] ppnew_adjlists_i[i];
    }
    delete[] ppnew_lvl2_nbs;
    delete[] ppnew_adjlists_o;
    delete[] ppnew_adjlists_i;

	// update number of vertices in graph
	hg.number_of_vertices = number_of_candidates;
    hg.number_of_edges = number_of_edges;
    hg.number_of_lvl2adj = number_of_lvl2adj;
}

// DEBUG - rm
// 3091 7267 7310 7382 7752 7761 8687 9050 9763 11140 12052 16755 17169 19191 21060 22582

void h_expand_level(CPU_Graph& hg, CPU_Data& hd, CPU_Cliques& hc, DS_Sizes& dss, 
                    int* minimum_out_degrees, int* minimum_in_degrees, 
                    double minimum_out_degree_ratio, double minimum_in_degree_ratio, 
                    int minimum_clique_size)
{
    uint64_t* read_count;              // read and write to tasks 1 and 2 in alternating manner
    uint64_t* read_offsets;
    Vertex* read_vertices;
    uint64_t* write_count;
    uint64_t* write_offsets;
    Vertex* write_vertices;
    uint64_t start;                    // old vertices information
    uint64_t end;
    int tot_vert;
    int num_mem;
    int num_cand;
    int expansions;
    int number_of_covered;
    Vertex* vertices;                   // new vertices information
    int number_of_members;
    int number_of_candidates;
    int total_vertices;
    int min_ext_out_deg;                    // calculate lower-upper bounds
    int min_ext_in_deg;
    int lower_bound;
    int upper_bound;
    int success;                  // helper
    int index;

    if ((*hd.current_level) % 2 == 0) {
        read_count = hd.tasks1_count;
        read_offsets = hd.tasks1_offset;
        read_vertices = hd.tasks1_vertices;
        write_count = hd.tasks2_count;
        write_offsets = hd.tasks2_offset;
        write_vertices = hd.tasks2_vertices;
    }
    else {
        read_count = hd.tasks2_count;
        read_offsets = hd.tasks2_offset;
        read_vertices = hd.tasks2_vertices;
        write_count = hd.tasks1_count;
        write_offsets = hd.tasks1_offset;
        write_vertices = hd.tasks1_vertices;
    }

    *write_count = 0;
    write_offsets[0] = 0;

    // --- CURRENT LEVEL ---

    for (int i = 0; i < *read_count; i++)
    {

        // INITIALIZE OLD VERTICES
        // get information of vertices being handled within tasks
        start = read_offsets[i];
        end = read_offsets[i + 1];
        tot_vert = end - start;
        num_mem = 0;
        for (uint64_t j = start; j < end; j++) {
            if (read_vertices[j].label != 1) {
                break;
            }
            num_mem++;
        }
        number_of_covered = 0;
        for (uint64_t j = start + num_mem; j < end; j++) {
            if (read_vertices[j].label != 2) {
                break;
            }
            number_of_covered++;
        }
        num_cand = tot_vert - num_mem;
        expansions = num_cand;

        // // LOOKAHEAD PRUNING
        // success = true;

        // // sets success to false if lookahead fails
        // h_lookahead_pruning(hg, hc, hd, read_vertices, tot_vert, num_mem, num_cand, start, 
        //                     minimum_out_degrees, minimum_in_degrees, minimum_clique_size, 
        //                     success);
        
        // if (success) {
        //     continue;
        // }

        // --- NEXT LEVEL ---

        for (int j = number_of_covered; j < expansions; j++) {

            // REMOVE ONE VERTEX
            if (j != number_of_covered) {
                success = true;

                // sets success to false is failed vertex found, sets to 2 if next vertex to be
                // added is a failed vertex
                h_remove_one_vertex(hg, hd, read_vertices, tot_vert, num_cand, num_mem, start, 
                                    minimum_out_degrees, minimum_in_degrees, minimum_clique_size,
                                    success);
                
                if (!success) {
                    break;
                }
                if(success == 2){
                    continue;
                }
            }

            // INITIALIZE NEW VERTICES
            vertices = new Vertex[tot_vert];
            number_of_members = num_mem;
            number_of_candidates = num_cand;
            total_vertices = tot_vert;
            
            for (index = 0; index < number_of_members; index++) {
                vertices[index] = read_vertices[start + index];
            }
            vertices[number_of_members] = read_vertices[start + total_vertices - 1];
            for (; index < total_vertices - 1; index++) {
                vertices[index + 1] = read_vertices[start + index];
            }

            if (number_of_covered > 0) {
                // set all covered vertices from previous level as candidates
                for (int j = num_mem + 1; j <= num_mem + number_of_covered; j++) {
                    vertices[j].label = 0;
                }
            }

            // ADD ONE VERTEX
            success = true;

            // sets success to false if failed found
            h_add_one_vertex(hg, hd, vertices, total_vertices, number_of_candidates, 
                             number_of_members, upper_bound, lower_bound, min_ext_out_deg, 
                             min_ext_in_deg, minimum_out_degrees, minimum_in_degrees, 
                             minimum_out_degree_ratio, minimum_in_degree_ratio, 
                             minimum_clique_size, success);

            // // if vertex in x found as not extendable, check if current set is clique and continue 
            // // to next iteration
            // if (!success) {
            //     // only first process needs to check and write clique as all processes do same
            //     if (grank == 0) {
            //         // check if current set is clique
            //         h_check_for_clique(hc, vertices, number_of_members, minimum_out_degrees, 
            //                        minimum_in_degrees, minimum_clique_size);
            //     }

            //     // continue to next iteration
            //     delete vertices;
            //     continue;
            // }

            // // CRITICAL VERTEX PRUNING
            // success = 1;

            // // sets success as 2 if critical fail, 0 if failed found or invalid bound, 1 otherwise
            // // h_critical_vertex_pruning(hg, hd, vertices, total_vertices, number_of_candidates, 
            // //                           number_of_members, upper_bound, lower_bound, min_ext_out_deg, 
            // //                           min_ext_in_deg, minimum_out_degrees, minimum_in_degrees, 
            // //                           minimum_out_degree_ratio, minimum_in_degree_ratio, 
            // //                           minimum_clique_size, success);

            // // if critical fail continue onto next iteration
            // if (success == 2) {
            //     delete vertices;
            //     continue;
            // }

            // CHECK FOR CLIQUE
            // only first process needs to check and write clique as all processes do same
            if (grank == 0) {
                // check if current set is clique
                h_check_for_clique(hc, vertices, number_of_members, minimum_out_degrees, 
                                   minimum_in_degrees, minimum_clique_size);
            }

            // continue to next iteration
            if (success == 0) {
                delete vertices;
                continue;
            }

            // WRITE TO TASKS
            // sort vertices so that lowest degree vertices are first in enumeration order before 
            // writing to tasks
            qsort(vertices, total_vertices, sizeof(Vertex), h_comp_vert_Q);

            if (number_of_candidates > 0) {
                h_write_to_tasks(hd, vertices, total_vertices, write_vertices, write_offsets, 
                                 write_count);
            }

            delete vertices;
        }
    }
}

// distributes work amongst processes in strided manner
void h_move_to_gpu(CPU_Data& hd, GPU_Data& h_dd, DS_Sizes& dss, string output)
{       
    uint64_t* tasks_count;          // read vertices information
    uint64_t* tasks_offset;
    Vertex* tasks_vertices;         
    uint64_t offset_start;
    uint64_t count;

    // get proper read location for level
    if(CPU_LEVELS % 2 == 0){
        tasks_count = hd.tasks2_count;
        tasks_offset = hd.tasks2_offset;
        tasks_vertices = hd.tasks2_vertices;
    }
    else{
        tasks_count = hd.tasks1_count;
        tasks_offset = hd.tasks1_offset;
        tasks_vertices = hd.tasks1_vertices;
    }

    // each process is assigned tasks in a strided manner, this step condenses those tasks
    count = *tasks_count;
    *tasks_count = 0;
    offset_start = 0;
    for(int i = grank; i < count; i += wsize){
        // increment assigned tasks count
        (*tasks_count)++;

        // copy vertices before offets are changed
        for(int j = tasks_offset[i]; j < tasks_offset[i + 1]; j++){
            tasks_vertices[offset_start + j - tasks_offset[i]] = tasks_vertices[j];
        }

        // copy offset and adjust
        tasks_offset[*tasks_count] = tasks_offset[i + 1] - tasks_offset[i] + offset_start;
        offset_start = tasks_offset[*tasks_count];
    }

    // each process is assigned tasks in a strided manner, this step condenses those tasks
    count = *hd.buffer_count;
    *hd.buffer_count = 0;
    offset_start = 0;
    for(int i = grank; i < count; i += wsize){
        // increment assigned tasks count
        (*hd.buffer_count)++;

        // copy vertices before offets are changed
        for(int j = hd.buffer_offset[i]; j < hd.buffer_offset[i + 1]; j++){
            hd.buffer_vertices[offset_start + j - hd.buffer_offset[i]] = hd.buffer_vertices[j];
        }

        // copy offset and adjust
        hd.buffer_offset[*hd.buffer_count] = hd.buffer_offset[i + 1] - hd.buffer_offset[i] + offset_start;
        offset_start = hd.buffer_offset[*hd.buffer_count];
    }

    // condense tasks
    h_fill_from_buffer(hd, dss.EXPAND_THRESHOLD);

    // move to GPU
    chkerr(cudaMemcpy(h_dd.tasks_count, tasks_count, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.tasks_offset, tasks_offset, (*tasks_count + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.tasks_vertices, tasks_vertices, tasks_offset[*tasks_count] * sizeof(Vertex), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.buffer_count, hd.buffer_count, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.buffer_offset, hd.buffer_offset, (*hd.buffer_count + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.buffer_vertices, hd.buffer_vertices, hd.buffer_offset[*hd.buffer_count] * sizeof(Vertex), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.current_level, hd.current_level, sizeof(uint64_t), cudaMemcpyHostToDevice));

    // DEBUG
    if(dss.DEBUG_TOGGLE){
        output_file << "GPU START" << endl;
        print_D_Data_Sizes(h_dd, dss);
    }
}

// move cliques from device to host
void h_dump_cliques(CPU_Cliques& hc, GPU_Data& h_dd, ofstream& temp_results, DS_Sizes& dss)
{
    // gpu cliques to cpu cliques
    chkerr(cudaMemcpy(hc.cliques_count, h_dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(hc.cliques_offset, h_dd.cliques_offset, sizeof(uint64_t) * (*hc.cliques_count + 1), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(hc.cliques_vertex, h_dd.cliques_vertex, sizeof(int) * hc.cliques_offset[*hc.cliques_count], cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    h_flush_cliques(hc, temp_results);

    cudaMemset(h_dd.cliques_count, 0, sizeof(uint64_t));
}

// move cliques from host to file
void h_flush_cliques(CPU_Cliques& hc, ofstream& temp_results) 
{
    uint64_t start;         // start of current clique
    uint64_t end;           // end of current clique

    for (int i = 0; i < ((*hc.cliques_count)); i++) {
        start = hc.cliques_offset[i];
        end = hc.cliques_offset[i + 1];
        temp_results << end - start << " ";
        for (uint64_t j = start; j < end; j++) {
            temp_results << hc.cliques_vertex[j] << " ";
        }
        temp_results << "\n";
    }
    ((*hc.cliques_count)) = 0;
}

void h_free_memory(CPU_Data& hd, GPU_Data& h_dd, CPU_Cliques& hc)
{
    // CPU DATA
    delete hd.tasks1_count;
    delete[] hd.tasks1_offset;
    delete[] hd.tasks1_vertices;
    delete hd.tasks2_count;
    delete[] hd.tasks2_offset;
    delete[] hd.tasks2_vertices;
    delete hd.buffer_count;
    delete[] hd.buffer_offset;
    delete[] hd.buffer_vertices;
    delete hd.current_level;
    delete hd.maximal_expansion;
    delete[] hd.vertex_order_map;
    delete[] hd.remaining_candidates;
    delete hd.remaining_count;
    delete[] hd.removed_candidates;
    delete hd.removed_count;
    delete[] hd.candidate_out_mem_degs;
    delete[] hd.candidate_in_mem_degs;
    // CPU CLIQUES
    delete hc.cliques_count;
    delete[] hc.cliques_vertex;
    delete[] hc.cliques_offset;
    // GPU GRAPH
    chkerr(cudaFree(h_dd.number_of_vertices));
    chkerr(cudaFree(h_dd.number_of_edges));
    chkerr(cudaFree(h_dd.out_neighbors));
    chkerr(cudaFree(h_dd.out_offsets));
    chkerr(cudaFree(h_dd.in_neighbors));
    chkerr(cudaFree(h_dd.in_offsets));
    chkerr(cudaFree(h_dd.twohop_neighbors));
    chkerr(cudaFree(h_dd.twohop_offsets));
    // GPU DATA
    chkerr(cudaFree(h_dd.current_level));
    chkerr(cudaFree(h_dd.tasks_count));
    chkerr(cudaFree(h_dd.tasks_offset));
    chkerr(cudaFree(h_dd.tasks_vertices));
    chkerr(cudaFree(h_dd.buffer_count));
    chkerr(cudaFree(h_dd.buffer_offset));
    chkerr(cudaFree(h_dd.buffer_vertices));
    chkerr(cudaFree(h_dd.wtasks_count));
    chkerr(cudaFree(h_dd.wtasks_offset));
    chkerr(cudaFree(h_dd.wtasks_vertices));
    chkerr(cudaFree(h_dd.global_vertices));
    chkerr(cudaFree(h_dd.remaining_candidates));
    chkerr(cudaFree(h_dd.lane_remaining_candidates));
    chkerr(cudaFree(h_dd.temp_vertex_array));
    chkerr(cudaFree(h_dd.removed_candidates));
    chkerr(cudaFree(h_dd.lane_removed_candidates));
    chkerr(cudaFree(h_dd.candidate_out_mem_degs));
    chkerr(cudaFree(h_dd.lane_candidate_out_mem_degs));
    chkerr(cudaFree(h_dd.candidate_in_mem_degs));
    chkerr(cudaFree(h_dd.lane_candidate_in_mem_degs));
    chkerr(cudaFree(h_dd.adjacencies));
    chkerr(cudaFree(h_dd.minimum_out_degree_ratio));
    chkerr(cudaFree(h_dd.minimum_out_degrees));
    chkerr(cudaFree(h_dd.minimum_in_degree_ratio));
    chkerr(cudaFree(h_dd.minimum_in_degrees));
    chkerr(cudaFree(h_dd.minimum_clique_size));
    chkerr(cudaFree(h_dd.total_tasks));
    chkerr(cudaFree(h_dd.current_task));
    chkerr(cudaFree(h_dd.vertex_order_map));
    // GPU CLIQUES
    chkerr(cudaFree(h_dd.cliques_count));
    chkerr(cudaFree(h_dd.cliques_vertex));
    chkerr(cudaFree(h_dd.cliques_offset));
    chkerr(cudaFree(h_dd.wcliques_count));
    chkerr(cudaFree(h_dd.wcliques_vertex));
    chkerr(cudaFree(h_dd.wcliques_offset));
    chkerr(cudaFree(h_dd.buffer_offset_start));
    chkerr(cudaFree(h_dd.buffer_start));
    chkerr(cudaFree(h_dd.cliques_offset_start));
    chkerr(cudaFree(h_dd.cliques_start));
    // DATA STRUCTURE SIZES
    chkerr(cudaFree(h_dd.TASKS_SIZE));
    chkerr(cudaFree(h_dd.TASKS_PER_WARP));
    chkerr(cudaFree(h_dd.BUFFER_SIZE));
    chkerr(cudaFree(h_dd.BUFFER_OFFSET_SIZE));
    chkerr(cudaFree(h_dd.CLIQUES_SIZE));
    chkerr(cudaFree(h_dd.CLIQUES_OFFSET_SIZE));
    chkerr(cudaFree(h_dd.CLIQUES_PERCENT));
    chkerr(cudaFree(h_dd.WCLIQUES_SIZE));
    chkerr(cudaFree(h_dd.WCLIQUES_OFFSET_SIZE));
    chkerr(cudaFree(h_dd.WTASKS_SIZE));
    chkerr(cudaFree(h_dd.WTASKS_OFFSET_SIZE));
    chkerr(cudaFree(h_dd.WVERTICES_SIZE));
    chkerr(cudaFree(h_dd.EXPAND_THRESHOLD));
    chkerr(cudaFree(h_dd.CLIQUES_DUMP));
}

// --- SECONDARY EXPANSION FUNCTIONS ---

// sets success to false if lookahead fails
void h_lookahead_pruning(CPU_Graph& hg, CPU_Cliques& hc, CPU_Data& hd, Vertex* read_vertices, 
                        int tot_vert, int num_mem, int num_cand, uint64_t start, 
                        int* minimum_out_degrees, int* minimum_in_degrees, int minimum_clique_size, 
                        int& success)
{
    uint64_t start_write;               // starting write position for new cliques
    int min_out_deg;
    int min_in_deg;
    uint64_t pneighbors_start;          
    uint64_t pneighbors_end;
    int phelper1;
    int pvertexid;

    min_out_deg = h_get_mindeg(tot_vert, minimum_out_degrees, minimum_clique_size);
    min_in_deg = h_get_mindeg(tot_vert, minimum_in_degrees, minimum_clique_size);

    // check if members meet degree requirement, dont need to check 2hop adj as diameter pruning 
    // guarentees all members will be within 2hops of eveything
    for (int i = 0; i < num_mem; i++) {
        if (read_vertices[start + i].out_mem_deg + read_vertices[start + i].out_can_deg < 
            min_out_deg || read_vertices[start + i].in_mem_deg + 
            read_vertices[start + i].in_can_deg < min_in_deg) {

            success = false;
            return;
        }
    }

    // initialize vertex order map
    for (int i = 0; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = i;
    }

    // update lvl2adj to candidates for all vertices
    for (int i = num_mem; i < tot_vert; i++) {
        read_vertices[start + i].lvl2adj = 0;
    }

    for (int i = num_mem; i < tot_vert; i++) {

        pvertexid = read_vertices[start + i].vertexid;

        pneighbors_start = hg.twohop_offsets[pvertexid];
        pneighbors_end = hg.twohop_offsets[pvertexid + 1];

        for (int j = pneighbors_start; j < pneighbors_end; j++) {

            phelper1 = hd.vertex_order_map[hg.twohop_neighbors[j]];

            if (phelper1 >= num_mem) {
                read_vertices[start + phelper1].lvl2adj++;
            }
        }
    }

    // reset vertex order map
    for (int i = 0; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = -1;
    }

    // check for lookahead
    for (int i = num_mem; i < tot_vert; i++) {
        if (read_vertices[start + i].lvl2adj < num_cand - 1 || read_vertices[start + i].out_mem_deg 
            + read_vertices[start + i].out_can_deg < min_out_deg || 
            read_vertices[start + i].in_mem_deg + read_vertices[start + i].in_can_deg < 
            min_in_deg) {

            success = false;
            return;
        }
    }

    if(grank == 0){
        // if we havent returned by this point lookahead pruning has succeded and we can write
        // write to cliques
        start_write = hc.cliques_offset[(*hc.cliques_count)];
        for (int i = 0; i < tot_vert; i++) {
            hc.cliques_vertex[start_write + i] = read_vertices[start + i].vertexid;
        }
        (*hc.cliques_count)++;
        hc.cliques_offset[(*hc.cliques_count)] = start_write + tot_vert;
    }
}

// sets success to false is failed vertex found 
void h_remove_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* read_vertices, int& tot_vert, 
                        int& num_cand, int& num_mem, uint64_t start, int* minimum_out_degrees, 
                        int* minimum_in_degrees, int minimum_clique_size, int& success)
{
    int pvertexid;                      // intersection
    uint64_t pneighbors_start;          
    uint64_t pneighbors_end;
    int phelper1;
    int min_out_deg;                    // helper variables
    int min_in_deg;

    min_out_deg = h_get_mindeg(num_mem + 1, minimum_out_degrees, minimum_clique_size);
    min_in_deg = h_get_mindeg(num_mem + 1, minimum_in_degrees, minimum_clique_size);

    // remove one vertex
    num_cand--;
    tot_vert--;

    // initialize vertex order map
    for (int i = 0; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = i;
    }

    // update info of vertices connected to removed cand
    pvertexid = read_vertices[start + tot_vert].vertexid;

    pneighbors_start = hg.out_offsets[pvertexid];
    pneighbors_end = hg.out_offsets[pvertexid + 1];

    for (int i = pneighbors_start; i < pneighbors_end; i++) {

        phelper1 = hd.vertex_order_map[hg.out_neighbors[i]];

        if (phelper1 > -1) {
            read_vertices[start + phelper1].in_can_deg--;

            if (read_vertices[start + phelper1].in_mem_deg + 
                read_vertices[start + phelper1].in_can_deg < min_in_deg) {
                
                if(phelper1 < num_mem){
                    success = false;
                    break;
                }
                else if(phelper1 == tot_vert - 1){
                    success = 2;
                }
            }
        }
    }

    // return if failed found
    if(!success){
        // reset vertex order map
        for (int i = 0; i < tot_vert; i++) {
            hd.vertex_order_map[read_vertices[start + i].vertexid] = -1;
        }

        return;
    }

    pneighbors_start = hg.in_offsets[pvertexid];
    pneighbors_end = hg.in_offsets[pvertexid + 1];

    for (int i = pneighbors_start; i < pneighbors_end; i++) {

        phelper1 = hd.vertex_order_map[hg.in_neighbors[i]];

        if (phelper1 > -1) {
            read_vertices[start + phelper1].out_can_deg--;

            if (read_vertices[start + phelper1].out_mem_deg + 
                read_vertices[start + phelper1].out_can_deg < min_out_deg) {
                
                if(phelper1 < num_mem){
                    success = false;
                    break;
                }
                else if(phelper1 == tot_vert - 1){
                    success = 2;
                }
            }
        }
    }

    // reset vertex order map
    for (int i = 0; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = -1;
    }
}

// sets success to false if failed found
void h_add_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, 
                     int& number_of_candidates, int& number_of_members, int& upper_bound, 
                     int& lower_bound, int& min_ext_out_deg, int& min_ext_in_deg, 
                     int* minimum_out_degrees, int* minimum_in_degrees, 
                     double minimum_out_degree_ratio, double minimum_in_degree_ratio, 
                     int minimum_clique_size, int& success)
{
    int pvertexid;                      // intersection
    uint64_t pneighbors_start;          
    uint64_t pneighbors_end;
    int phelper1;
    int min_out_deg;
    int min_in_deg;

    min_out_deg = h_get_mindeg(number_of_members + 2, minimum_out_degrees, minimum_clique_size);
    min_in_deg = h_get_mindeg(number_of_members + 2, minimum_in_degrees, minimum_clique_size);

    // initialize vertex order map
    for (int i = 0; i < total_vertices; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = i;
    }

    // ADD ONE VERTEX
    pvertexid = vertices[number_of_members].vertexid;

    vertices[number_of_members].label = 1;
    number_of_members++;
    number_of_candidates--;

    // update adjacencies of newly added vertex
    pneighbors_start = hg.out_offsets[pvertexid];
    pneighbors_end = hg.out_offsets[pvertexid + 1];

    for (int i = pneighbors_start; i < pneighbors_end; i++) {

        phelper1 = hd.vertex_order_map[hg.out_neighbors[i]];

        if (phelper1 > -1) {
            vertices[phelper1].in_mem_deg++;
            vertices[phelper1].in_can_deg--;
        }
    }

    pneighbors_start = hg.in_offsets[pvertexid];
    pneighbors_end = hg.in_offsets[pvertexid + 1];

    for (int i = pneighbors_start; i < pneighbors_end; i++) {

        phelper1 = hd.vertex_order_map[hg.in_neighbors[i]];

        if (phelper1 > -1) {
            vertices[phelper1].out_mem_deg++;
            vertices[phelper1].out_can_deg--;
        }
    }

    // DIAMETER PRUNING
    h_diameter_pruning(hg, hd, vertices, pvertexid, total_vertices, number_of_candidates, 
                       number_of_members, min_out_deg, min_in_deg);

    // DEGREE-BASED PRUNING
    // sets success to false if failed found else leaves as true
    h_degree_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, 
                     upper_bound, lower_bound, min_ext_out_deg, min_ext_in_deg, 
                     minimum_out_degrees, minimum_in_degrees, minimum_out_degree_ratio, 
                     minimum_in_degree_ratio, minimum_clique_size, success);
}

// sets success as 2 if critical fail, 0 if failed found or invalid bound, 1 otherwise
void h_critical_vertex_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, 
                              int& number_of_candidates, int& number_of_members, int& upper_bound, 
                              int& lower_bound, int& min_ext_out_deg, int& min_ext_in_deg, 
                              int* minimum_out_degrees, int* minimum_in_degrees, 
                              double minimum_out_degree_ratio, double minimum_in_degree_ratio, 
                              int minimum_clique_size, int& success)
{
    int pvertexid;                      // intersection
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;
    bool critical_fail;                 // helper
    int number_of_crit_adj;
    int* adj_counters;

    // initialize vertex order map
    for (int i = 0; i < total_vertices; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = i;
    }

    // CRITICAL VERTEX PRUNING
    // iterate through all vertices in clique
    for (int k = 0; k < number_of_members; k++)
    {
        pvertexid = vertices[k].vertexid;

        // if they are a critical vertex
        if (vertices[k].out_mem_deg + vertices[k].out_can_deg == 
            minimum_out_degrees[number_of_members + lower_bound] && vertices[k].out_can_deg > 0) {

            // iterate through all neighbors
            pneighbors_start = hg.out_offsets[pvertexid];
            pneighbors_end = hg.out_offsets[pvertexid + 1];

            for (uint64_t l = pneighbors_start; l < pneighbors_end; l++) {

                phelper1 = hd.vertex_order_map[hg.out_neighbors[l]];

                // if neighbor is cand
                if (phelper1 >= number_of_members) {
                    vertices[phelper1].label = 4;
                }
            }
        }

        // if they are a critical vertex
        if (vertices[k].in_mem_deg + vertices[k].in_can_deg == 
            minimum_in_degrees[number_of_members + lower_bound] && vertices[k].in_can_deg > 0) {

            // iterate through all neighbors
            pneighbors_start = hg.in_offsets[pvertexid];
            pneighbors_end = hg.in_offsets[pvertexid + 1];

            for (uint64_t l = pneighbors_start; l < pneighbors_end; l++) {

                phelper1 = hd.vertex_order_map[hg.in_neighbors[l]];

                // if neighbor is cand
                if (phelper1 >= number_of_members) {
                    vertices[phelper1].label = 4;
                }
            }
        }
    }

    // reset vertex order map
    for (int i = 0; i < total_vertices; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = -1;
    }

    // sort vertices so that critical vertex adjacent candidates are immediately after vertices 
    // within the clique
    qsort(vertices + number_of_members, number_of_candidates, sizeof(Vertex), h_comp_vert_cv);

    // calculate number of critical adjacent vertices
    number_of_crit_adj = 0;
    for (int i = number_of_members; i < total_vertices; i++) {
        if (vertices[i].label == 4) {
            number_of_crit_adj++;
        }
        else {
            break;
        }
    }

    // if there were no neighbors of critical vertices stop here
    if (number_of_crit_adj == 0)
    {
        return;
    }

    // adj_counter[0] = 10, means that the vertex at position 0 in new_vertices has 10 critical 
    // vertices neighbors within 2 hops
    adj_counters = new int[total_vertices];

    // initialize vertex order map
    for (int i = 0; i < total_vertices; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = i;
        adj_counters[i] = 0;
    }

    // calculate adj_counters, adjacencies to critical vertices
    for (int i = number_of_members; i < number_of_members + number_of_crit_adj; i++) {

        pvertexid = vertices[i].vertexid;

        // track 2hop adj
        pneighbors_start = hg.twohop_offsets[pvertexid];
        pneighbors_end = hg.twohop_offsets[pvertexid + 1];

        for (uint64_t k = pneighbors_start; k < pneighbors_end; k++) {

            phelper1 = hd.vertex_order_map[hg.twohop_neighbors[k]];

            if (phelper1 > -1) {
                adj_counters[phelper1]++;
            }
        }
    }

    // check for critical failure
    critical_fail = false;

    // all vertices within the clique must be within 2hops of the newly ah_dded critical vertex adj vertices
    for (int k = 0; k < number_of_members; k++) {
        if (adj_counters[k] != number_of_crit_adj) {
            critical_fail = true;
            break;
        }
    }

    if (critical_fail) {
        // reset vertex order map
        for (int i = 0; i < total_vertices; i++) {
            hd.vertex_order_map[vertices[i].vertexid] = -1;
        }

        delete adj_counters;
        success = 2;
        return;
    }

    // all critical adj vertices must all be within 2 hops of each other
    for (int k = number_of_members; k < number_of_members + number_of_crit_adj; k++) {
        if (adj_counters[k] < number_of_crit_adj - 1) {
            critical_fail = true;
            break;
        }
    }

    if (critical_fail) {
        // reset vertex order map
        for (int i = 0; i < total_vertices; i++) {
            hd.vertex_order_map[vertices[i].vertexid] = -1;
        }

        delete adj_counters;
        success = 2;
        return;
    }

    // iterate through all critical adjacent
    for (int i = number_of_members; i < number_of_members + number_of_crit_adj; i++) {

        pvertexid = vertices[i].vertexid;

        // update 1hop adj
        pneighbors_start = hg.out_offsets[pvertexid];
        pneighbors_end = hg.out_offsets[pvertexid + 1];

        for (uint64_t k = pneighbors_start; k < pneighbors_end; k++) {

            phelper1 = hd.vertex_order_map[hg.out_neighbors[k]];

            if (phelper1 > -1) {
                vertices[phelper1].in_mem_deg++;
                vertices[phelper1].in_can_deg--;
            }
        }

        pneighbors_start = hg.in_offsets[pvertexid];
        pneighbors_end = hg.in_offsets[pvertexid + 1];

        for (uint64_t k = pneighbors_start; k < pneighbors_end; k++) {

            phelper1 = hd.vertex_order_map[hg.in_neighbors[k]];

            if (phelper1 > -1) {
                vertices[phelper1].out_mem_deg++;
                vertices[phelper1].out_can_deg--;
            }
        }
    }

    // no failed vertices found so add all critical vertex adj candidates to clique
    for (int k = number_of_members; k < number_of_members + number_of_crit_adj; k++) {
        vertices[k].label = 1;
    }
    number_of_members += number_of_crit_adj;
    number_of_candidates -= number_of_crit_adj;

    // DIAMTER PRUNING
    h_diameter_pruning_cv(hd, vertices, total_vertices, number_of_members, adj_counters, 
                          number_of_crit_adj);

    delete adj_counters;

    // DEGREE-BASED PRUNING
    // sets success to false if failed found else leaves as true
    h_degree_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, 
                     upper_bound, lower_bound, min_ext_out_deg, min_ext_in_deg, 
                     minimum_out_degrees, minimum_in_degrees, minimum_out_degree_ratio, 
                     minimum_in_degree_ratio, minimum_clique_size, success);
}

void h_diameter_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int pvertexid, 
                        int& total_vertices, int& number_of_candidates, int number_of_members, 
                        int min_out_deg, int min_in_deg)
{
    uint64_t pneighbors_start;          // intersection
    uint64_t pneighbors_end;
    int phelper1;

    (*hd.remaining_count) = 0;

    // set all candidates as invalid
    for (int i = number_of_members; i < total_vertices; i++) {
        vertices[i].label = -1;
    }

    // iterate through all lvl2adj of added vertex and validate all neighbor candidates
    pneighbors_start = hg.twohop_offsets[pvertexid];
    pneighbors_end = hg.twohop_offsets[pvertexid + 1];

    for (int i = pneighbors_start; i < pneighbors_end; i++) {

        phelper1 = hd.vertex_order_map[hg.twohop_neighbors[i]];

        // if neighbor is a candidate
        if (phelper1 >= number_of_members) {
            vertices[phelper1].label = 0;

            // only track mem degs of candidates which pass basic degree pruning
            if(vertices[phelper1].out_mem_deg + vertices[phelper1].out_can_deg >= min_out_deg && 
               vertices[phelper1].in_mem_deg + vertices[phelper1].in_can_deg >= min_in_deg){
                
                hd.candidate_out_mem_degs[(*hd.remaining_count)] = vertices[phelper1].out_mem_deg;
                hd.candidate_in_mem_degs[(*hd.remaining_count)] = vertices[phelper1].in_mem_deg;
                (*hd.remaining_count)++;
            }
        }
    }
}

void h_diameter_pruning_cv(CPU_Data& hd, Vertex* vertices, int& total_vertices, 
                           int number_of_members, int* adj_counters, int number_of_crit_adj)
{
    *hd.remaining_count = 0;

    // remove all cands who are not within 2hops of all newly added cands
    for (int k = number_of_members; k < total_vertices; k++) {

        if (adj_counters[k] == number_of_crit_adj) {

            hd.candidate_out_mem_degs[*hd.remaining_count] = vertices[k].out_mem_deg;
            hd.candidate_in_mem_degs[*hd.remaining_count] = vertices[k].in_mem_deg;
            (*hd.remaining_count)++;
        }
        else {
            vertices[k].label = -1;
        }
    }
}

// sets success to false if failed found else leaves as true
void h_degree_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, 
                      int& number_of_candidates, int number_of_members, int& upper_bound, 
                      int& lower_bound, int& min_ext_out_deg, int& min_ext_in_deg, 
                      int* minimum_out_degrees, int* minimum_in_degrees, 
                      double minimum_out_degree_ratio, double minimum_in_degree_ratio, 
                      int minimum_clique_size, int& success)
{
    int pvertexid;                      // intersection
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;
    int num_val_cands;                  // helper variables

    // used for bound calculation
    qsort(hd.candidate_out_mem_degs, (*hd.remaining_count), sizeof(int), h_comp_int_desc);
    qsort(hd.candidate_in_mem_degs, (*hd.remaining_count), sizeof(int), h_comp_int_desc);

    // set bounds and min ext degs
    h_calculate_LU_bounds(hd, upper_bound, lower_bound, min_ext_out_deg, min_ext_in_deg, vertices, 
                          number_of_members, *hd.remaining_count, minimum_out_degrees, 
                          minimum_in_degrees, minimum_out_degree_ratio, minimum_in_degree_ratio, 
                          minimum_clique_size, success);

    // check whether new bounds are valid
    if(success == false){
        for (int i = 0; i < hg.number_of_vertices; i++) {
            hd.vertex_order_map[i] = -1;
        }
        return;
    }

    // check for failed vertices
    for (int k = 0; k < number_of_members; k++) {
        if (!h_vert_isextendable(vertices[k], number_of_members, upper_bound, lower_bound, 
                                    min_ext_out_deg, min_ext_in_deg, minimum_out_degrees, 
                                    minimum_in_degrees, minimum_clique_size)) {

            success = false;
            for (int i = 0; i < hg.number_of_vertices; i++) {
                hd.vertex_order_map[i] = -1;
            }
            return;
        }
    }
    

    (*hd.remaining_count) = 0;
    (*hd.removed_count) = 0;

    // check for invalid candidates
    for (int i = number_of_members; i < total_vertices; i++) {
        if (vertices[i].label == 0 && 
            h_cand_isvalid(vertices[i], number_of_members, upper_bound, lower_bound, min_ext_out_deg, min_ext_in_deg, 
                           minimum_out_degrees, minimum_in_degrees, minimum_clique_size)) {
            
            hd.remaining_candidates[(*hd.remaining_count)++] = i;
        }
        else {
            hd.removed_candidates[(*hd.removed_count)++] = i;
        }
    }

    // while some vertices were pruned but not all vertices have been pruned
    while (*hd.remaining_count > 0 && *hd.removed_count > 0) { 
        
        // update degrees
        if ((*hd.remaining_count) < (*hd.removed_count)) {
            
            // reset can degs
            for (int i = 0; i < total_vertices; i++) {
                vertices[i].in_can_deg = 0;
                vertices[i].out_can_deg = 0;
            }

            // update degrees for all vertices
            for (int i = 0; i < *hd.remaining_count; i++) {

                pvertexid = vertices[hd.remaining_candidates[i]].vertexid;

                // update using out degs
                pneighbors_start = hg.out_offsets[pvertexid];
                pneighbors_end = hg.out_offsets[pvertexid + 1];

                for (int j = pneighbors_start; j < pneighbors_end; j++) {

                    phelper1 = hd.vertex_order_map[hg.out_neighbors[j]];

                    if (phelper1 > -1) {
                        vertices[phelper1].in_can_deg++;
                    }
                }

                // update using in degs
                pneighbors_start = hg.in_offsets[pvertexid];
                pneighbors_end = hg.in_offsets[pvertexid + 1];

                for (int j = pneighbors_start; j < pneighbors_end; j++) {

                    phelper1 = hd.vertex_order_map[hg.in_neighbors[j]];

                    if (phelper1 > -1) {
                        vertices[phelper1].out_can_deg++;
                    }
                }
            }
        }
        else {

            for (int i = 0; i < *hd.removed_count; i++) {

                pvertexid = vertices[hd.removed_candidates[i]].vertexid;

                pneighbors_start = hg.out_offsets[pvertexid];
                pneighbors_end = hg.out_offsets[pvertexid + 1];

                for (int j = pneighbors_start; j < pneighbors_end; j++) {

                    phelper1 = hd.vertex_order_map[hg.out_neighbors[j]];

                    if (phelper1 > -1) {
                        vertices[phelper1].in_can_deg--;
                    }
                }

                pneighbors_start = hg.in_offsets[pvertexid];
                pneighbors_end = hg.in_offsets[pvertexid + 1];

                for (int j = pneighbors_start; j < pneighbors_end; j++) {

                    phelper1 = hd.vertex_order_map[hg.in_neighbors[j]];

                    if (phelper1 > -1) {
                        vertices[phelper1].out_can_deg--;
                    }
                }
            }
        }

        num_val_cands = 0;

        for (int k = 0; k < (*hd.remaining_count); k++) {
            if (h_cand_isvalid(vertices[hd.remaining_candidates[k]], number_of_members, 
                               upper_bound, lower_bound, min_ext_out_deg, min_ext_in_deg, minimum_out_degrees, 
                               minimum_in_degrees, minimum_clique_size)) {

                hd.candidate_out_mem_degs[num_val_cands] = 
                    vertices[hd.remaining_candidates[k]].out_mem_deg;
                hd.candidate_in_mem_degs[num_val_cands] = 
                    vertices[hd.remaining_candidates[k]].in_mem_deg;
                num_val_cands++;
            }
        }

        qsort(hd.candidate_out_mem_degs, num_val_cands, sizeof(int), h_comp_int_desc);
        qsort(hd.candidate_in_mem_degs, num_val_cands, sizeof(int), h_comp_int_desc);

        // set bounds and min ext degs
        h_calculate_LU_bounds(hd, upper_bound, lower_bound, min_ext_out_deg, min_ext_in_deg, vertices, 
                            number_of_members, num_val_cands, minimum_out_degrees, 
                            minimum_in_degrees, minimum_out_degree_ratio, minimum_in_degree_ratio, 
                            minimum_clique_size, success);

        // check whether new bounds are valid
        if(success == false){    
            for (int i = 0; i < hg.number_of_vertices; i++) {
                hd.vertex_order_map[i] = -1;
            }
            return;
        }

        // check for failed vertices
        for (int k = 0; k < number_of_members; k++) {
            if (!h_vert_isextendable(vertices[k], number_of_members, upper_bound, lower_bound, 
                                     min_ext_out_deg, min_ext_in_deg, minimum_out_degrees, 
                                     minimum_in_degrees, minimum_clique_size)) {

                success = false;
                for (int i = 0; i < hg.number_of_vertices; i++) {
                    hd.vertex_order_map[i] = -1;
                }
                return;
            }
        }

        num_val_cands = 0;
        (*hd.removed_count) = 0;

        // check for invalid candidates
        for (int k = 0; k < (*hd.remaining_count); k++) {
            if (h_cand_isvalid(vertices[hd.remaining_candidates[k]], number_of_members, 
                               upper_bound, lower_bound, min_ext_out_deg, min_ext_in_deg, minimum_out_degrees, 
                               minimum_in_degrees, minimum_clique_size)) {
                
                hd.remaining_candidates[num_val_cands++] = hd.remaining_candidates[k];
            }
            else {
                hd.removed_candidates[(*hd.removed_count)++] = hd.remaining_candidates[k];
            }
        }

        (*hd.remaining_count) = num_val_cands;
    }

    for (int i = 0; i < hg.number_of_vertices; i++) {
        hd.vertex_order_map[i] = -1;
    }

    for (int i = 0; i < (*hd.remaining_count); i++) {
        vertices[number_of_members + i] = vertices[hd.remaining_candidates[i]];
    }

    total_vertices = total_vertices - number_of_candidates + (*hd.remaining_count);
    number_of_candidates = (*hd.remaining_count);
}

// sets bounds and min ext degs
void h_calculate_LU_bounds(CPU_Data& hd, int& upper_bound, int& lower_bound, int& min_ext_out_deg, 
                           int& min_ext_in_deg, Vertex* vertices, int number_of_members, 
                           int number_of_candidates, int* minimum_out_degrees, 
                           int* minimum_in_degrees, double minimum_out_degree_ratio, 
                           double minimum_in_degree_ratio, int minimum_clique_size, int& success)
{
    //lower & upper bound are initialized using the degree of vertex in S
	//and tighten using the degree of vertex in ext_S
	int i, ntightened_max_cands;
	int nmin_clq_clqdeg_o, nminclqdeg_candeg_o, nmin_clq_totaldeg_o, nclq_clqdeg_sum_o, ncand_clqdeg_sum_o;
	int nmin_clq_clqdeg_i, nminclqdeg_candeg_i, nmin_clq_totaldeg_i, nclq_clqdeg_sum_i, ncand_clqdeg_sum_i;

	//clq_clqdeg means: v in S (clq) 's indegree (clqdeg)
	nmin_clq_clqdeg_o = vertices[0].out_mem_deg;
	nminclqdeg_candeg_o = vertices[0].out_can_deg;
	nclq_clqdeg_sum_o = vertices[0].out_mem_deg;
	nmin_clq_totaldeg_o = vertices[0].out_mem_deg+vertices[0].out_can_deg;

	nmin_clq_clqdeg_i = vertices[0].in_mem_deg;
	nminclqdeg_candeg_i = vertices[0].in_can_deg;
	nclq_clqdeg_sum_i = vertices[0].in_mem_deg;
	nmin_clq_totaldeg_i = vertices[0].in_mem_deg+vertices[0].in_can_deg;

	for(i=1;i<number_of_members;i++)
	{
		// out direction
		nclq_clqdeg_sum_o += vertices[i].out_mem_deg;
		if(nmin_clq_clqdeg_o>vertices[i].out_mem_deg)
		{
			nmin_clq_clqdeg_o = vertices[i].out_mem_deg;
			nminclqdeg_candeg_o = vertices[i].out_can_deg;
		}
		else if(nmin_clq_clqdeg_o==vertices[i].out_mem_deg)
		{
			if(nminclqdeg_candeg_o>vertices[i].out_can_deg){
				nminclqdeg_candeg_o = vertices[i].out_can_deg;
            }
		}

		if(nmin_clq_totaldeg_o>vertices[i].out_mem_deg+vertices[i].out_can_deg){
			nmin_clq_totaldeg_o = vertices[i].out_mem_deg+vertices[i].out_can_deg;
        }

		// in direction
		nclq_clqdeg_sum_i += vertices[i].in_mem_deg;
		if(nmin_clq_clqdeg_i>vertices[i].in_mem_deg)
		{
			nmin_clq_clqdeg_i = vertices[i].in_mem_deg;
			nminclqdeg_candeg_i = vertices[i].in_can_deg;
		}
		else if(nmin_clq_clqdeg_i==vertices[i].in_mem_deg)
		{
			if(nminclqdeg_candeg_i>vertices[i].in_can_deg){
				nminclqdeg_candeg_i = vertices[i].in_can_deg;
            }
		}

		if(nmin_clq_totaldeg_i>vertices[i].in_mem_deg+vertices[i].in_can_deg){
			nmin_clq_totaldeg_i = vertices[i].in_mem_deg+vertices[i].in_can_deg;
        }
	}

	min_ext_out_deg = h_get_mindeg(number_of_members+1, minimum_out_degrees, minimum_clique_size);
	min_ext_in_deg = h_get_mindeg(number_of_members+1, minimum_in_degrees, minimum_clique_size);
	
    if(nmin_clq_clqdeg_o<minimum_out_degrees[number_of_members] || nmin_clq_clqdeg_i<minimum_in_degrees[number_of_members])//check the requirment of bound pruning rule
	{
		// ==== calculate L_min and U_min ====
		//initialize lower bound
		int nmin_cands = max((h_get_mindeg(number_of_members, minimum_out_degrees, minimum_clique_size)-nmin_clq_clqdeg_o),
				(h_get_mindeg(number_of_members, minimum_in_degrees, minimum_clique_size)-nmin_clq_clqdeg_i));
		int nmin_cands_o = nmin_cands;

		while(nmin_cands_o<=nminclqdeg_candeg_o && nmin_clq_clqdeg_o+nmin_cands_o<minimum_out_degrees[number_of_members+nmin_cands_o]){
			nmin_cands_o++;
        }

		if(nmin_clq_clqdeg_o+nmin_cands_o<minimum_out_degrees[number_of_members+nmin_cands_o]){
			nmin_cands_o = number_of_candidates+1;
            success = false;
            return;
        }

		int nmin_cands_i = nmin_cands;

		while(nmin_cands_i<=nminclqdeg_candeg_i && nmin_clq_clqdeg_i+nmin_cands_i<minimum_in_degrees[number_of_members+nmin_cands_i]){
			nmin_cands_i++;
        }

		if(nmin_clq_clqdeg_i+nmin_cands_i<minimum_in_degrees[number_of_members+nmin_cands_i]){
			nmin_cands_i = number_of_candidates+1;
            success = false;
            return;
        }

		lower_bound = max(nmin_cands_o, nmin_cands_i);

		//initialize upper bound
		upper_bound = min((int)(nmin_clq_totaldeg_o/minimum_out_degree_ratio),
				(int)(nmin_clq_totaldeg_i/minimum_in_degree_ratio))+1-number_of_members;

		if(upper_bound>number_of_candidates){
			upper_bound = number_of_candidates;
        }

		// ==== tighten lower bound and upper bound based on the clique degree of candidates ====
		if(lower_bound<upper_bound)
		{
			//tighten lower bound
			ncand_clqdeg_sum_o = 0;
			ncand_clqdeg_sum_i = 0;

			for(i=0;i<lower_bound;i++)
			{
				ncand_clqdeg_sum_o += hd.candidate_out_mem_degs[i];
				ncand_clqdeg_sum_i += hd.candidate_in_mem_degs[i];
			}

			while(i<upper_bound
					&& nclq_clqdeg_sum_o+ncand_clqdeg_sum_i<number_of_members*minimum_out_degrees[number_of_members+i]
					&& nclq_clqdeg_sum_i+ncand_clqdeg_sum_o<number_of_members*minimum_in_degrees[number_of_members+i])
			{
				ncand_clqdeg_sum_o += hd.candidate_out_mem_degs[i];
				ncand_clqdeg_sum_i += hd.candidate_in_mem_degs[i];
				i++;
			}

			if(nclq_clqdeg_sum_o+ncand_clqdeg_sum_o<number_of_members*minimum_out_degrees[number_of_members+i]
				&& nclq_clqdeg_sum_i+ncand_clqdeg_sum_i<number_of_members*minimum_in_degrees[number_of_members+i]){
				lower_bound = upper_bound+1;
                success = false;
                return;
            }
			else //tighten upper bound
			{
				lower_bound = i;
				ntightened_max_cands = i;

				while(i<upper_bound)
				{
					ncand_clqdeg_sum_o += hd.candidate_out_mem_degs[i];
					ncand_clqdeg_sum_i += hd.candidate_in_mem_degs[i];
					i++;
					if(nclq_clqdeg_sum_o+ncand_clqdeg_sum_i>=number_of_members*minimum_out_degrees[number_of_members+i]
						&& nclq_clqdeg_sum_i+ncand_clqdeg_sum_o>=number_of_members*minimum_in_degrees[number_of_members+i])
						ntightened_max_cands = i;
				}

				if(upper_bound>ntightened_max_cands){
					upper_bound = ntightened_max_cands;
                }

				if(lower_bound>1)
				{
					min_ext_out_deg = h_get_mindeg(number_of_members+lower_bound, minimum_out_degrees, minimum_clique_size);
					min_ext_in_deg = h_get_mindeg(number_of_members+lower_bound, minimum_in_degrees, minimum_clique_size);
				}
			}
		}
	}
	else
	{
		upper_bound = number_of_candidates;
		
        if(number_of_members<minimum_clique_size){
			lower_bound = minimum_clique_size-number_of_members;
        }
		else{
			lower_bound = 0;
        }
	}

	if(number_of_members+upper_bound<minimum_clique_size){
        success = false;
        return;
    }

	if (upper_bound < 0 || upper_bound < lower_bound) {
        success = false;
    }
}

void h_check_for_clique(CPU_Cliques& hc, Vertex* vertices, int number_of_members, 
                        int* minimum_out_degrees, int* minimum_in_degrees, int minimum_clique_size)
{
    bool clique;
    int min_out_deg;
    int min_in_deg;

    if (number_of_members < minimum_clique_size) {
        return;
    }

    clique = true;

    min_out_deg = minimum_out_degrees[number_of_members];
    min_in_deg = minimum_in_degrees[number_of_members];

    for (int k = 0; k < number_of_members; k++) {
        if (vertices[k].out_mem_deg < min_out_deg || vertices[k].in_mem_deg < min_in_deg) {
            clique = false;
            break;
        }
    }

    // if clique write to cliques array
    if (clique) {
        uint64_t start_write = hc.cliques_offset[(*hc.cliques_count)];
        for (int k = 0; k < number_of_members; k++) {
            hc.cliques_vertex[start_write + k] = vertices[k].vertexid;
        }
        (*hc.cliques_count)++;
        hc.cliques_offset[(*hc.cliques_count)] = start_write + number_of_members;
    }
}

void h_write_to_tasks(CPU_Data& hd, Vertex* vertices, int total_vertices, Vertex* write_vertices, 
                      uint64_t* write_offsets, uint64_t* write_count)
{
    if ((*write_count) < CPU_EXPAND_THRESHOLD) {
        uint64_t start_write = write_offsets[*write_count];

        for (int k = 0; k < total_vertices; k++) {
            write_vertices[start_write + k] = vertices[k];
        }

        (*write_count)++;
        write_offsets[*write_count] = start_write + total_vertices;
    }
    else {
        uint64_t start_write = hd.buffer_offset[(*hd.buffer_count)];

        for (int k = 0; k < total_vertices; k++) {
            hd.buffer_vertices[start_write + k] = vertices[k];
        }

        (*hd.buffer_count)++;
        hd.buffer_offset[(*hd.buffer_count)] = start_write + total_vertices;
    }
}

void h_fill_from_buffer(CPU_Data& hd, int threshold)
{
    int write_amount;
    uint64_t start_buffer;
    uint64_t end_buffer;
    uint64_t size_buffer;
    uint64_t start_write;
    uint64_t* write_count;
    uint64_t* write_offsets;
    Vertex* write_vertices;

    if ((*hd.current_level) % 2 == 0) {
        write_count = hd.tasks2_count;
        write_offsets = hd.tasks2_offset;
        write_vertices = hd.tasks2_vertices;
    }
    else {
        write_count = hd.tasks1_count;
        write_offsets = hd.tasks1_offset;
        write_vertices = hd.tasks1_vertices;
    }

    // get read and write locations
    write_amount = ((*hd.buffer_count) >= (threshold - *write_count)) ? threshold - *write_count : (*hd.buffer_count);
    start_buffer = hd.buffer_offset[(*hd.buffer_count) - write_amount];
    end_buffer = hd.buffer_offset[(*hd.buffer_count)];
    size_buffer = end_buffer - start_buffer;
    start_write = write_offsets[*write_count];

    // copy tasks data from end of buffer to end of tasks
    memcpy(&write_vertices[start_write], &hd.buffer_vertices[start_buffer], sizeof(Vertex) * size_buffer);

    // handle offsets
    for (int j = 1; j <= write_amount; j++) {
        write_offsets[*write_count + j] = start_write + (hd.buffer_offset[(*hd.buffer_count) - write_amount + j] - start_buffer);
    }

    // update counts
    (*write_count) += write_amount;
    (*hd.buffer_count) -= write_amount;
}