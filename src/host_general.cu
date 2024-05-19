#include "../inc/common.h"
#include "../inc/host_general.h"
#include "../inc/host_expansion.h"
#include "../inc/host_helper.h"
#include "../inc/host_debug.h"
#include "../inc/device_general.h"

// initializes minimum degrees array 
void calculate_minimum_degrees(CPU_Graph& hg)
{
    minimum_degrees = new int[hg.number_of_vertices + 1];
    minimum_degrees[0] = 0;
    for (int i = 1; i <= hg.number_of_vertices; i++) {
        minimum_degrees[i] = ceil(minimum_degree_ratio * (i - 1));
    }
}

void search(CPU_Graph& hg, ofstream& temp_results, ofstream& output_file) 
{
    // DATA STRUCTURES
    CPU_Data hd;
    CPU_Cliques hc;
    GPU_Data dd;



    // HANDLE MEMORY
    allocate_memory(hd, dd, hc, hg);
    cudaDeviceSynchronize();



    // TIME
    auto start = chrono::high_resolution_clock::now();



    // INITIALIZE TASKS
    if(grank == 0){
        cout << ">:INITIALIZING TASKS" << endl;
    }
    initialize_tasks(hg, hd);



    // DEBUG
    if (DEBUG_TOGGLE) {
        mvs = (*(hd.tasks1_offset + (*hd.tasks1_count)));
        if ((*(hd.tasks1_offset + (*hd.tasks1_count))) > WVERTICES_SIZE) {
            cout << "!!! VERTICES SIZE ERROR !!!" << endl;
            return;
        }
        h_print_Data_Sizes(hd, hc, output_file);
    }



    // CPU EXPANSION
    // cpu levels is multiplied by two to ensure that data ends up in tasks1, this allows us to always copy tasks1 without worry like before hybrid cpu approach
    // cpu expand must be called atleast one time to handle first round cover pruning as the gpu code cannot do this
    for (int i = 0; i < CPU_LEVELS + 1 && !(*hd.maximal_expansion); i++) {
        h_expand_level(hg, hd, hc);
    
        // if cliques is more than threshold dump
        if (hc.cliques_offset[(*hc.cliques_count)] > CLIQUES_DUMP) {
            flush_cliques(hc, temp_results);
        }



        // DEBUG
        if (DEBUG_TOGGLE) {
            h_print_Data_Sizes(hd, hc, output_file);
        }
    }

    flush_cliques(hc, temp_results);



    // TODO - divide work and move to cpu, how should this be done?
    // TODO - for now doing block shift, change later
    // TODO - remove CPU MODE as it is not used anyways
    // TRANSFER TO GPU
    if (!CPU_MODE) {
        move_to_gpu(hd, dd);
        cudaDeviceSynchronize();
    }



    // TODO - use cuTS distributed loop
    // EXPAND LEVEL
    if(grank == 0){
        cout << ">:BEGINNING EXPANSION" << endl;
    }
    while (!(*hd.maximal_expansion))
    {
        (*(hd.maximal_expansion)) = true;
        chkerr(cudaMemset(dd.current_task, 0, sizeof(int)));
        cudaDeviceSynchronize();

        // expand all tasks in 'tasks' array, each warp will write to their respective warp tasks buffer in global memory
        d_expand_level<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
        cudaDeviceSynchronize();



        // DEBUG
        if (DEBUG_TOGGLE) {
            if (print_Warp_Data_Sizes_Every(dd, 1, output_file)) { break; }
        }



        // consolidate all the warp tasks/cliques buffers into the next global tasks array, buffer, and cliques
        transfer_buffers<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
        cudaDeviceSynchronize();



        // determine whether maximal expansion has been accomplished
        uint64_t current_level, write_count, buffer_count;
        chkerr(cudaMemcpy(&current_level, dd.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(&buffer_count, dd.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        if (current_level % 2 == 0) {
            chkerr(cudaMemcpy(&write_count, dd.tasks2_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        }
        else {
            chkerr(cudaMemcpy(&write_count, dd.tasks1_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        }

        if (write_count > 0 || buffer_count > 0) {
            (*(hd.maximal_expansion)) = false;
        }

        // TODO - what number of tasks is enough to split



        chkerr(cudaMemset(dd.wtasks_count, 0, sizeof(uint64_t) * NUMBER_OF_WARPS));
        chkerr(cudaMemset(dd.wcliques_count, 0, sizeof(uint64_t) * NUMBER_OF_WARPS));
        if (write_count < EXPAND_THRESHOLD && buffer_count > 0) {
            // if not enough tasks were generated when expanding the previous level to fill the next tasks array the program will attempt to fill the tasks array by popping tasks from the buffer
            fill_from_buffer<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
            cudaDeviceSynchronize();
        }
        current_level++;
        chkerr(cudaMemcpy(dd.current_level, &current_level, sizeof(uint64_t), cudaMemcpyHostToDevice));




        // determine whether cliques has exceeded defined threshold, if so dump them to a file
        uint64_t cliques_size, cliques_count;
        chkerr(cudaMemcpy(&cliques_count, dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(&cliques_size, dd.cliques_offset + cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        // if cliques is more than threshold dump
        if (cliques_size > CLIQUES_DUMP) {
            dump_cliques(hc, dd, temp_results);
        }



        // DEBUG
        if (DEBUG_TOGGLE) {
            if (print_Data_Sizes_Every(dd, 1, output_file)) { break; }
        }
    }



    // TIME
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    MPI_Barrier(MPI_COMM_WORLD);
    if(grank == 0){
        cout << "--->:ENUMERATION TIME: " << duration.count() << " ms" << endl;
    }



    dump_cliques(hc, dd, temp_results);

    free_memory(hd, dd, hc);
}

// allocates memory for the data structures on the host and device   
void allocate_memory(CPU_Data& hd, GPU_Data& dd, CPU_Cliques& hc, CPU_Graph& hg)
{
    // GPU GRAPH
    chkerr(cudaMalloc((void**)&dd.number_of_vertices, sizeof(int)));
    chkerr(cudaMalloc((void**)&dd.number_of_edges, sizeof(int)));
    chkerr(cudaMalloc((void**)&dd.onehop_neighbors, sizeof(int) * hg.number_of_edges));
    chkerr(cudaMalloc((void**)&dd.onehop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&dd.twohop_neighbors, sizeof(int) * hg.number_of_lvl2adj));
    chkerr(cudaMalloc((void**)&dd.twohop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1)));

    chkerr(cudaMemcpy(dd.number_of_vertices, &(hg.number_of_vertices), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.number_of_edges, &(hg.number_of_edges), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.onehop_neighbors, hg.onehop_neighbors, sizeof(int) * hg.number_of_edges, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.onehop_offsets, hg.onehop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.twohop_neighbors, hg.twohop_neighbors, sizeof(int) * hg.number_of_lvl2adj, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.twohop_offsets, hg.twohop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));

    // CPU DATA
    hd.tasks1_count = new uint64_t;
    hd.tasks1_offset = new uint64_t[EXPAND_THRESHOLD + 1];
    hd.tasks1_vertices = new Vertex[TASKS_SIZE];

    hd.tasks1_offset[0] = 0;
    (*(hd.tasks1_count)) = 0;

    hd.tasks2_count = new uint64_t;
    hd.tasks2_offset = new uint64_t[EXPAND_THRESHOLD + 1];
    hd.tasks2_vertices = new Vertex[TASKS_SIZE];

    hd.tasks2_offset[0] = 0;
    (*(hd.tasks2_count)) = 0;

    hd.buffer_count = new uint64_t;
    hd.buffer_offset = new uint64_t[BUFFER_OFFSET_SIZE];
    hd.buffer_vertices = new Vertex[BUFFER_SIZE];

    hd.buffer_offset[0] = 0;
    (*(hd.buffer_count)) = 0;

    hd.current_level = new uint64_t;
    hd.maximal_expansion = new bool;
    hd.dumping_cliques = new bool;

    (*hd.current_level) = 0;
    (*hd.maximal_expansion) = false;
    (*hd.dumping_cliques) = false;

    hd.vertex_order_map = new int[hg.number_of_vertices];
    hd.remaining_candidates = new int[hg.number_of_vertices];
    hd.removed_candidates = new int[hg.number_of_vertices];
    hd.remaining_count = new int;
    hd.removed_count = new int;
    hd.candidate_indegs = new int[hg.number_of_vertices];

    memset(hd.vertex_order_map, -1, sizeof(int) * hg.number_of_vertices);

    // GPU DATA
    chkerr(cudaMalloc((void**)&dd.current_level, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&dd.tasks1_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.tasks1_offset, sizeof(uint64_t) * (EXPAND_THRESHOLD + 1)));
    chkerr(cudaMalloc((void**)&dd.tasks1_vertices, sizeof(Vertex) * TASKS_SIZE));

    chkerr(cudaMemset(dd.tasks1_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(dd.tasks1_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&dd.tasks2_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.tasks2_offset, sizeof(uint64_t) * (EXPAND_THRESHOLD + 1)));
    chkerr(cudaMalloc((void**)&dd.tasks2_vertices, sizeof(Vertex) * TASKS_SIZE));

    chkerr(cudaMemset(dd.tasks2_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(dd.tasks2_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&dd.buffer_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.buffer_offset, sizeof(uint64_t) * BUFFER_OFFSET_SIZE));
    chkerr(cudaMalloc((void**)&dd.buffer_vertices, sizeof(Vertex) * BUFFER_SIZE));

    chkerr(cudaMemset(dd.buffer_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(dd.buffer_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&dd.wtasks_count, sizeof(uint64_t) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&dd.wtasks_offset, (sizeof(uint64_t) * WTASKS_OFFSET_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&dd.wtasks_vertices, (sizeof(Vertex) * WTASKS_SIZE) * NUMBER_OF_WARPS));

    chkerr(cudaMemset(dd.wtasks_offset, 0, (sizeof(uint64_t) * WTASKS_OFFSET_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMemset(dd.wtasks_count, 0, sizeof(uint64_t) * NUMBER_OF_WARPS));

    chkerr(cudaMalloc((void**)&dd.global_vertices, (sizeof(Vertex) * WVERTICES_SIZE) * NUMBER_OF_WARPS));

    chkerr(cudaMalloc((void**)&dd.removed_candidates, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&dd.lane_removed_candidates, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));

    chkerr(cudaMalloc((void**)&dd.remaining_candidates, (sizeof(Vertex) * WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&dd.lane_remaining_candidates, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));

    chkerr(cudaMalloc((void**)&dd.candidate_indegs, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&dd.lane_candidate_indegs, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));

    chkerr(cudaMalloc((void**)&dd.adjacencies, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));

    chkerr(cudaMalloc((void**)&dd.minimum_degree_ratio, sizeof(double)));
    chkerr(cudaMalloc((void**)&dd.minimum_degrees, sizeof(int) * (hg.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&dd.minimum_clique_size, sizeof(int)));
    chkerr(cudaMalloc((void**)&dd.scheduling_toggle, sizeof(int)));

    chkerr(cudaMemcpy(dd.minimum_degree_ratio, &minimum_degree_ratio, sizeof(double), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.minimum_degrees, minimum_degrees, sizeof(int) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.minimum_clique_size, &minimum_clique_size, sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.scheduling_toggle, &scheduling_toggle, sizeof(int), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void**)&dd.total_tasks, sizeof(int)));

    chkerr(cudaMemset(dd.total_tasks, 0, sizeof(int)));

    // CPU CLIQUES
    hc.cliques_count = new uint64_t;
    hc.cliques_vertex = new int[CLIQUES_SIZE];
    hc.cliques_offset = new uint64_t[CLIQUES_OFFSET_SIZE];

    hc.cliques_offset[0] = 0;
    (*(hc.cliques_count)) = 0;

    // GPU CLIQUES
    chkerr(cudaMalloc((void**)&dd.cliques_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.cliques_vertex, sizeof(int) * CLIQUES_SIZE));
    chkerr(cudaMalloc((void**)&dd.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE));

    chkerr(cudaMemset(dd.cliques_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(dd.cliques_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&dd.wcliques_count, sizeof(uint64_t) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&dd.wcliques_offset, (sizeof(uint64_t) * WCLIQUES_OFFSET_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&dd.wcliques_vertex, (sizeof(int) * WCLIQUES_SIZE) * NUMBER_OF_WARPS));

    chkerr(cudaMemset(dd.wcliques_offset, 0, (sizeof(uint64_t) * WCLIQUES_OFFSET_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMemset(dd.wcliques_count, 0, sizeof(uint64_t) * NUMBER_OF_WARPS));

    chkerr(cudaMalloc((void**)&dd.total_cliques, sizeof(int)));

    chkerr(cudaMemset(dd.total_cliques, 0, sizeof(int)));

    chkerr(cudaMalloc((void**)&dd.buffer_offset_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.buffer_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.cliques_offset_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.cliques_start, sizeof(uint64_t)));

    // task scheduling
    chkerr(cudaMalloc((void**)&dd.current_task, sizeof(int)));
}

// processes 0th level of expansion
void initialize_tasks(CPU_Graph& hg, CPU_Data& hd)
{
    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;

    // cover pruning
    int maximum_degree;
    int maximum_degree_index;

    // vertices information
    int total_vertices;
    int number_of_candidates;
    Vertex* vertices;



    (*hd.remaining_count) = 0;
    (*hd.removed_count) = 0;

    // initialize vertices
    total_vertices = hg.number_of_vertices;
    vertices = new Vertex[total_vertices];
    number_of_candidates = total_vertices;
    for (int i = 0; i < total_vertices; i++) {
        vertices[i].vertexid = i;
        vertices[i].indeg = 0;
        vertices[i].exdeg = hg.onehop_offsets[i + 1] - hg.onehop_offsets[i];
        vertices[i].lvl2adj = hg.twohop_offsets[i + 1] - hg.twohop_offsets[i];
        if (vertices[i].exdeg >= minimum_degrees[minimum_clique_size] && vertices[i].lvl2adj >= minimum_clique_size - 1) {
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
            vertices[hd.remaining_candidates[i]].exdeg = 0;
        }

        for (int i = 0; i < number_of_candidates; i++) {
            // in 0th level id is same as position in vertices as all vertices are in vertices, see last block
            pvertexid = hd.remaining_candidates[i];
            pneighbors_start = hg.onehop_offsets[pvertexid];
            pneighbors_end = hg.onehop_offsets[pvertexid + 1];
            for (int j = pneighbors_start; j < pneighbors_end; j++) {
                phelper1 = hg.onehop_neighbors[j];
                if (vertices[phelper1].label == 0) {
                    vertices[phelper1].exdeg++;
                }
            }
        }

        (*hd.remaining_count) = 0;
        (*hd.removed_count) = 0;

        // remove more vertices based on updated degrees
        for (int i = 0; i < number_of_candidates; i++) {
            phelper1 = hd.remaining_candidates[i];
            if (vertices[phelper1].exdeg >= minimum_degrees[minimum_clique_size]) {
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
    int removed_start = 0;
    while((*hd.removed_count) > removed_start) {
        pvertexid = hd.removed_candidates[removed_start];
        pneighbors_start = hg.onehop_offsets[pvertexid];
        pneighbors_end = hg.onehop_offsets[pvertexid + 1];

        for (int j = pneighbors_start; j < pneighbors_end; j++) {
            phelper1 = hg.onehop_neighbors[j];

            if (vertices[phelper1].label == 0) {
                vertices[phelper1].exdeg--;

                if (vertices[phelper1].exdeg < minimum_degrees[minimum_clique_size]) {
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
            if (vertices[i].exdeg > maximum_degree) {
                maximum_degree = vertices[i].exdeg;
                maximum_degree_index = i;
            }
        }
    }
    vertices[maximum_degree_index].label = 3;

    // find all covered vertices
    pneighbors_start = hg.onehop_offsets[maximum_degree_index];
    pneighbors_end = hg.onehop_offsets[maximum_degree_index + 1];
    for (int i = pneighbors_start; i < pneighbors_end; i++) {
        pvertexid = hg.onehop_neighbors[i];
        if (vertices[pvertexid].label == 0) {
            vertices[pvertexid].label = 2;
        }
    }

    // sort enumeration order before writing to tasks
    qsort(vertices, total_vertices, sizeof(Vertex), h_sort_vert_Q);
    total_vertices = number_of_candidates;



    // WRITE TO TASKS
    if (total_vertices > 0)
    {
        for (int j = 0; j < total_vertices; j++) {
            hd.tasks1_vertices[j].vertexid = vertices[j].vertexid;
            hd.tasks1_vertices[j].label = vertices[j].label;
            hd.tasks1_vertices[j].indeg = vertices[j].indeg;
            hd.tasks1_vertices[j].exdeg = vertices[j].exdeg;
            hd.tasks1_vertices[j].lvl2adj = 0;
        }
        (*(hd.tasks1_count))++;
        hd.tasks1_offset[(*(hd.tasks1_count))] = total_vertices;
    }

    delete vertices;
}

void h_expand_level(CPU_Graph& hg, CPU_Data& hd, CPU_Cliques& hc)
{
    // initiate the variables containing the location of the read and write task vectors, done in an alternating, odd-even manner like the c-intersection of cuTS
    uint64_t* read_count;
    uint64_t* read_offsets;
    Vertex* read_vertices;
    uint64_t* write_count;
    uint64_t* write_offsets;
    Vertex* write_vertices;

    // old vertices information
    uint64_t start;
    uint64_t end;
    int tot_vert;
    int num_mem;
    int num_cand;
    int expansions;
    int number_of_covered;

    // new vertices information
    Vertex* vertices;
    int number_of_members;
    int number_of_candidates;
    int total_vertices;

    // calculate lower-upper bounds
    int min_ext_deg;
    int lower_bound;
    int upper_bound;

    int method_return;
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

    // set to false later if task is generated indicating non-maximal expansion
    (*hd.maximal_expansion) = true;



    // CURRENT LEVEL
    for (int i = 0; i < *read_count; i++)
    {
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



        // LOOKAHEAD PRUNING
        method_return = h_lookahead_pruning(hg, hc, hd, read_vertices, tot_vert, num_mem, num_cand, start);
        if (method_return) {
            continue;
        }



        // NEXT LEVEL
        for (int j = number_of_covered; j < expansions; j++) {



            // REMOVE ONE VERTEX
            if (j != number_of_covered) {
                method_return = h_remove_one_vertex(hg, hd, read_vertices, tot_vert, num_cand, num_mem, start);
                if (method_return) {
                    break;
                }
            }



            // NEW VERTICES
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
            method_return = h_add_one_vertex(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

            // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
            if (method_return == 1) {
                if (number_of_members >= minimum_clique_size) {
                    h_check_for_clique(hc, vertices, number_of_members);
                }

                delete vertices;
                continue;
            }



            // CRITICAL VERTEX PRUNING
            method_return = h_critical_vertex_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

            // if critical fail continue onto next iteration
            if (method_return == 2) {
                delete vertices;
                continue;
            }



            // CHECK FOR CLIQUE
            // all processes will do this, to prevent duplicates only process 0 will save cpu results
            if (grank == 0 && number_of_members >= minimum_clique_size) {
                h_check_for_clique(hc, vertices, number_of_members);
            }

            // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
            if (method_return == 1) {
                delete vertices;
                continue;
            }



            // WRITE TO TASKS
            //sort vertices so that lowest degree vertices are first in enumeration order before writing to tasks
            qsort(vertices, total_vertices, sizeof(Vertex), h_sort_vert_Q);

            if (number_of_candidates > 0) {
                h_write_to_tasks(hd, vertices, total_vertices, write_vertices, write_offsets, write_count);
            }



            delete vertices;
        }
    }



    // FILL TASKS FROM BUFFER
    // if last CPU round copy enough tasks for GPU expansion
    if ((*hd.current_level) == CPU_LEVELS && CPU_EXPAND_THRESHOLD < EXPAND_THRESHOLD && (*hd.buffer_count) > 0) {
        h_fill_from_buffer(hd, write_vertices, write_offsets, write_count, EXPAND_THRESHOLD);
    }
    // if not enough generated to fully populate fill from buffer
    if (*write_count < CPU_EXPAND_THRESHOLD && (*hd.buffer_count) > 0){
        h_fill_from_buffer(hd, write_vertices, write_offsets, write_count, CPU_EXPAND_THRESHOLD);
    }

    (*hd.current_level)++;
}

// NEW - changed to distribute work amongst processes
void move_to_gpu(CPU_Data& hd, GPU_Data& dd)
{
    uint64_t* tasks_count;
    uint64_t* tasks_offset;
    Vertex* tasks_vertices;

    uint64_t block_size;
    uint64_t block_start;

    uint64_t offset_start;



    // split tasks
    // get proper read location for level
    if(CPU_LEVELS % 2 == 1){
        tasks_count = hd.tasks1_count;
        tasks_offset = hd.tasks1_offset;
        tasks_vertices = hd.tasks1_vertices;
    }
    else{
        tasks_count = hd.tasks2_count;
        tasks_offset = hd.tasks2_offset;
        tasks_vertices = hd.tasks2_vertices;
    }

    // get work size for tasks
    block_size = *tasks_count / NUMBER_OF_PROCESSESS;
    block_start = block_size * grank;
    if(grank == NUMBER_OF_PROCESSESS - 1){
        block_size += *tasks_count % NUMBER_OF_PROCESSESS;
    }

    // rearange tasks
    memmove(tasks_count, &block_size, sizeof(uint64_t));
    memmove(tasks_offset, tasks_offset + block_start, sizeof(uint64_t) * (block_size + 1));
    memmove(tasks_vertices, tasks_vertices + tasks_offset[0], sizeof(Vertex) * (tasks_offset[block_size] - tasks_offset[0]));

    // revalue tasks
    offset_start = tasks_offset[0];
    for(int i = 0; i <= block_size; i++){
        tasks_offset[i] -= offset_start;
    }

    // get work size for buffer
    block_size = *hd.buffer_count / NUMBER_OF_PROCESSESS;
    block_start = block_size * grank;
    if(grank == NUMBER_OF_PROCESSESS - 1){
        block_size += *hd.buffer_count % NUMBER_OF_PROCESSESS;
    }

    // rearange buffer
    memmove(hd.buffer_count, &block_size, sizeof(uint64_t));
    memmove(hd.buffer_offset, hd.buffer_offset + block_start, sizeof(uint64_t) * (block_size + 1));
    memmove(hd.buffer_vertices, hd.buffer_vertices + hd.buffer_offset[0], sizeof(Vertex) * (hd.buffer_offset[block_size] - hd.buffer_offset[0]));

    // revalue buffer
    offset_start = hd.buffer_offset[0];
    for(int i = 0; i <= block_size; i++){
        hd.buffer_offset[i] -= offset_start;
    }

    // condense tasks
    h_fill_from_buffer(hd, tasks_vertices, tasks_offset, tasks_count, EXPAND_THRESHOLD);

    // TODO - only copy whats needed
    // move to GPU
    if (CPU_LEVELS % 2 == 1) {
        chkerr(cudaMemcpy(dd.tasks1_count, hd.tasks1_count, sizeof(uint64_t), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.tasks1_offset, hd.tasks1_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.tasks1_vertices, hd.tasks1_vertices, (TASKS_SIZE) * sizeof(Vertex), cudaMemcpyHostToDevice));
    }
    else{
        chkerr(cudaMemcpy(dd.tasks2_count, hd.tasks2_count, sizeof(uint64_t), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.tasks2_offset, hd.tasks2_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.tasks2_vertices, hd.tasks2_vertices, (TASKS_SIZE) * sizeof(Vertex), cudaMemcpyHostToDevice));
    }

    chkerr(cudaMemcpy(dd.buffer_count, hd.buffer_count, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.buffer_offset, hd.buffer_offset, (BUFFER_OFFSET_SIZE) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.buffer_vertices, hd.buffer_vertices, (BUFFER_SIZE) * sizeof(int), cudaMemcpyHostToDevice));

    chkerr(cudaMemcpy(dd.current_level, hd.current_level, sizeof(uint64_t), cudaMemcpyHostToDevice));
}

void dump_cliques(CPU_Cliques& hc, GPU_Data& dd, ofstream& temp_results)
{
    // gpu cliques to cpu cliques
    chkerr(cudaMemcpy(hc.cliques_count, dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(hc.cliques_offset, dd.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(hc.cliques_vertex, dd.cliques_vertex, sizeof(int) * CLIQUES_SIZE, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // DEBUG
    //print_CPU_Cliques(hc);

    flush_cliques(hc, temp_results);

    cudaMemset(dd.cliques_count, 0, sizeof(uint64_t));
}

void flush_cliques(CPU_Cliques& hc, ofstream& temp_results) 
{
    for (int i = 0; i < ((*hc.cliques_count)); i++) {
        uint64_t start = hc.cliques_offset[i];
        uint64_t end = hc.cliques_offset[i + 1];
        temp_results << end - start << " ";
        for (uint64_t j = start; j < end; j++) {
            temp_results << hc.cliques_vertex[j] << " ";
        }
        temp_results << "\n";
    }
    ((*hc.cliques_count)) = 0;
}

void free_memory(CPU_Data& hd, GPU_Data& dd, CPU_Cliques& hc)
{
    // GPU GRAPH
    chkerr(cudaFree(dd.number_of_vertices));
    chkerr(cudaFree(dd.number_of_edges));
    chkerr(cudaFree(dd.onehop_neighbors));
    chkerr(cudaFree(dd.onehop_offsets));
    chkerr(cudaFree(dd.twohop_neighbors));
    chkerr(cudaFree(dd.twohop_offsets));

    // CPU DATA
    delete hd.tasks1_count;
    delete hd.tasks1_offset;
    delete hd.tasks1_vertices;

    delete hd.tasks2_count;
    delete hd.tasks2_offset;
    delete hd.tasks2_vertices;

    delete hd.buffer_count;
    delete hd.buffer_offset;
    delete hd.buffer_vertices;

    delete hd.current_level;
    delete hd.maximal_expansion;
    delete hd.dumping_cliques;

    delete hd.vertex_order_map;
    delete hd.remaining_candidates;
    delete hd.remaining_count;
    delete hd.removed_candidates;
    delete hd.removed_count;
    delete hd.candidate_indegs;

    // GPU DATA
    chkerr(cudaFree(dd.current_level));

    chkerr(cudaFree(dd.tasks1_count));
    chkerr(cudaFree(dd.tasks1_offset));
    chkerr(cudaFree(dd.tasks1_vertices));

    chkerr(cudaFree(dd.tasks2_count));
    chkerr(cudaFree(dd.tasks2_offset));
    chkerr(cudaFree(dd.tasks2_vertices));

    chkerr(cudaFree(dd.buffer_count));
    chkerr(cudaFree(dd.buffer_offset));
    chkerr(cudaFree(dd.buffer_vertices));

    chkerr(cudaFree(dd.wtasks_count));
    chkerr(cudaFree(dd.wtasks_offset));
    chkerr(cudaFree(dd.wtasks_vertices));

    chkerr(cudaFree(dd.global_vertices));

    chkerr(cudaFree(dd.remaining_candidates));
    chkerr(cudaFree(dd.lane_remaining_candidates));

    chkerr(cudaFree(dd.removed_candidates));
    chkerr(cudaFree(dd.lane_removed_candidates));

    chkerr(cudaFree(dd.candidate_indegs));
    chkerr(cudaFree(dd.lane_candidate_indegs));

    chkerr(cudaFree(dd.adjacencies));

    chkerr(cudaFree(dd.minimum_degree_ratio));
    chkerr(cudaFree(dd.minimum_degrees));
    chkerr(cudaFree(dd.minimum_clique_size));
    chkerr(cudaFree(dd.scheduling_toggle));

    chkerr(cudaFree(dd.total_tasks));

    // CPU CLIQUES
    delete hc.cliques_count;
    delete hc.cliques_vertex;
    delete hc.cliques_offset;

    // GPU CLIQUES
    chkerr(cudaFree(dd.cliques_count));
    chkerr(cudaFree(dd.cliques_vertex));
    chkerr(cudaFree(dd.cliques_offset));

    chkerr(cudaFree(dd.wcliques_count));
    chkerr(cudaFree(dd.wcliques_vertex));
    chkerr(cudaFree(dd.wcliques_offset));

    chkerr(cudaFree(dd.buffer_offset_start));
    chkerr(cudaFree(dd.buffer_start));
    chkerr(cudaFree(dd.cliques_offset_start));
    chkerr(cudaFree(dd.cliques_start));

    // tasks scheduling
    chkerr(cudaFree(dd.current_task));
}