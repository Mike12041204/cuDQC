#include "../inc/common.hpp"
#include "../inc/host_debug.h"

bool print_D_Data_Sizes(GPU_Data& h_dd, DS_Sizes& dss)
{
    uint64_t* current_level = new uint64_t;
    uint64_t* tasks1_count = new uint64_t;
    uint64_t* buffer_count = new uint64_t;
    uint64_t* cliques_count = new uint64_t;
    uint64_t* tasks1_size = new uint64_t;
    uint64_t* BUFFER_SIZE = new uint64_t;
    uint64_t* CLIQUES_SIZE = new uint64_t;

    chkerr(cudaMemcpy(current_level, h_dd.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_count, h_dd.tasks_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_count, h_dd.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_count, h_dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_size, h_dd.tasks_offset + (*tasks1_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(BUFFER_SIZE, h_dd.buffer_offset + (*buffer_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(CLIQUES_SIZE, h_dd.cliques_offset + (*cliques_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));

    output_file << "L: " << (*current_level) << " T: " << (*tasks1_count) << " " << (*tasks1_size) << " B: " << (*buffer_count) << " " << (*BUFFER_SIZE) << " C: " << (*cliques_count) << " " << (*CLIQUES_SIZE) << endl << endl;

    if (*tasks1_size > mts) {
        mts = *tasks1_size;
    }
    if (*BUFFER_SIZE > mbs) {
        mbs = *BUFFER_SIZE;
    }
    if (*buffer_count > mbo) {
        mbo = *buffer_count;
    }
    if (*CLIQUES_SIZE > mcs) {
        mcs = *CLIQUES_SIZE;
    }
    if (*cliques_count > mco) {
        mco = *cliques_count;
    }

    if ((*tasks1_count) > dss.EXPAND_THRESHOLD || (*tasks1_size) > dss.TASKS_SIZE || (*buffer_count) > dss.BUFFER_OFFSET_SIZE || (*BUFFER_SIZE) > dss.BUFFER_SIZE || (*cliques_count) > dss.CLIQUES_OFFSET_SIZE ||
        (*CLIQUES_SIZE) > dss.CLIQUES_SIZE) {
        output_file << "!!! GLOBAL STRUCTURE SIZE ERROR !!!" << endl;
        return true;
    }

    delete current_level;
    delete tasks1_count;
    delete buffer_count;
    delete cliques_count;
    delete tasks1_size;
    delete BUFFER_SIZE;
    delete CLIQUES_SIZE;
    
    return false;
}

// returns true if warp buffer was too small causing error
bool print_D_Warp_Data_Sizes(GPU_Data& h_dd, DS_Sizes& dss)
{
    uint64_t* tasks_counts = new uint64_t[NUMBER_OF_WARPS];
    uint64_t* TASKS_SIZEs = new uint64_t[NUMBER_OF_WARPS];
    int tasks_tcount = 0;
    int tasks_tsize = 0;
    int tasks_mcount = 0;
    int tasks_msize = 0;
    uint64_t* cliques_counts = new uint64_t[NUMBER_OF_WARPS];
    uint64_t* CLIQUES_SIZEs = new uint64_t[NUMBER_OF_WARPS];
    int cliques_tcount = 0;
    int cliques_tsize = 0;
    int cliques_mcount = 0;
    int cliques_msize = 0;

    chkerr(cudaMemcpy(tasks_counts, h_dd.wtasks_count, sizeof(uint64_t) * NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_counts, h_dd.wcliques_count, sizeof(uint64_t) * NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUMBER_OF_WARPS; i++) {
        chkerr(cudaMemcpy(TASKS_SIZEs + i, h_dd.wtasks_offset + (i * dss.WTASKS_OFFSET_SIZE) + tasks_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(CLIQUES_SIZEs + i, h_dd.wcliques_offset + (i * dss.WCLIQUES_OFFSET_SIZE) + cliques_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < NUMBER_OF_WARPS; i++) {
        tasks_tcount += tasks_counts[i];
        if (tasks_counts[i] > tasks_mcount) {
            tasks_mcount = tasks_counts[i];
        }
        tasks_tsize += TASKS_SIZEs[i];
        if (TASKS_SIZEs[i] > tasks_msize) {
            tasks_msize = TASKS_SIZEs[i];
        }
        cliques_tcount += cliques_counts[i];
        if (cliques_counts[i] > cliques_mcount) {
            cliques_mcount = cliques_counts[i];
        }
        cliques_tsize += CLIQUES_SIZEs[i];
        if (CLIQUES_SIZEs[i] > cliques_msize) {
            cliques_msize = CLIQUES_SIZEs[i];
        }
    }

    // DEBUG - if you need total sizes back you can uncomment this line, but they dont decide ds sizes so they arent needed in most cases
    //output_file << "WTasks( TC: " << tasks_tcount << " TS: " << tasks_tsize << " MC: " << tasks_mcount << " MS: " << tasks_msize << ") WCliques ( TC: " << cliques_tcount << " TS: " << cliques_tsize << " MC: " << cliques_mcount << " MS: " << cliques_msize << ")" << endl;
    output_file << "T: " << tasks_tcount << "(" << tasks_mcount << ") " << tasks_tsize << "(" << tasks_msize << ") C: " << cliques_tcount << "(" << cliques_mcount << ") " << cliques_tsize << "("<< cliques_msize << ")" << endl;

    if (tasks_mcount > wto) {
        wto = tasks_mcount;
    }
    if (tasks_msize > wts) {
        wts = tasks_msize;
    }
    if (cliques_mcount > wco) {
        wco = cliques_mcount;
    }
    if (cliques_msize > wcs) {
        wcs = cliques_msize;
    }

    if (tasks_mcount > dss.WTASKS_OFFSET_SIZE || tasks_msize > dss.WTASKS_SIZE || cliques_mcount > dss.WCLIQUES_OFFSET_SIZE || cliques_msize > dss.WCLIQUES_SIZE) {
        output_file << "!!! WARP STRUCTURE SIZE ERROR !!!" << endl;
        return true;
    }

    delete tasks_counts;
    delete TASKS_SIZEs;
    delete cliques_counts;
    delete CLIQUES_SIZEs;

    return false;
}

void print_H_Data_Sizes(CPU_Data& hd, CPU_Cliques& hc)
{
    output_file << "L: " << (*hd.current_level) << " T1: " << (*hd.tasks1_count) << " " << (*(hd.tasks1_offset + (*hd.tasks1_count))) << " T2: " << (*hd.tasks2_count) << " " << 
        (*(hd.tasks2_offset + (*hd.tasks2_count))) << " B: " << (*hd.buffer_count) << " " << (*(hd.buffer_offset + (*hd.buffer_count))) << " C: " << 
        (*hc.cliques_count) << " " << (*(hc.cliques_offset + (*hc.cliques_count))) << endl << endl;

    if ((*(hd.tasks1_offset + (*hd.tasks1_count))) > mts) {
        mts = (*(hd.tasks1_offset + (*hd.tasks1_count)));
    }
    if ((*(hd.tasks2_offset + (*hd.tasks2_count))) > mts) {
        mts = (*(hd.tasks2_offset + (*hd.tasks2_count)));
    }
    if ((*(hd.buffer_offset + (*hd.buffer_count))) > mbs) {
        mbs = (*(hd.buffer_offset + (*hd.buffer_count)));
    }
    if ((*hd.buffer_count) > mbo) {
        mbo = (*hd.buffer_count);
    }
    if ((*(hc.cliques_offset + (*hc.cliques_count))) > mcs) {
        mcs = (*(hc.cliques_offset + (*hc.cliques_count)));
    }
    if ((*hc.cliques_count) > mco) {
        mco = (*hc.cliques_count);
    }
}

void initialize_maxes()
{
    mts = 0;
    mbs = 0;
    mbo = 0;
    mcs = 0;
    mco = 0;
    wts = 0;
    wto = 0;
    wcs = 0;
    wco = 0;
    mvs = 0;
}

void print_maxes()
{
    output_file
        << "DATA STRUCTURE SIZES:" << endl
        << "TASKS SIZE: " << mts << endl
        << "BUFFER SIZE: " << mbs << endl
        << "BUFFER OFFSET SIZE: " << mbo << endl
        << "CLIQUES SIZE: " << mcs << endl
        << "CLIQUES OFFSET SIZE: " << mco << endl
        << "WCLIQUES SIZE: " << wcs << endl
        << "WCLIQUES OFFSET SIZE: " << wco << endl
        << "WTASKS SIZE: " << wts << endl
        << "WTASKS OFFSET SIZE: " << wto << endl
        << "WVERTICES SIZE: " << mvs << endl;
}

void print_graph(CPU_Graph& hg)
{
    output_file 
        << "|V| = " << hg.number_of_vertices 
        << " |E| = " << hg.number_of_edges 
        << " |2hop| = " << hg.number_of_lvl2adj << endl << endl;
}

// void print_CPU_Graph(CPU_Graph& hg) {
//     cout << endl << " --- (CPU_Graph)host_graph details --- " << endl;
//     cout << endl << "|V|: " << hg.number_of_vertices << " |E|: " << hg.number_of_edges << endl;
//     cout << endl << "Out Offsets:" << endl;
//     for (int i = 0; i <= hg.number_of_vertices; i++) {
//         cout << hg.out_offsets[i] << " ";
//     }
//     cout << endl << "Out Neighbors:" << endl;
//     for (int i = 0; i < hg.number_of_edges; i++) {
//         cout << hg.out_neighbors[i] << " ";
//     }
//     cout << endl << "In Offsets:" << endl;
//     for (int i = 0; i <= hg.number_of_vertices; i++) {
//         cout << hg.in_offsets[i] << " ";
//     }
//     cout << endl << "In Neighbors:" << endl;
//     for (int i = 0; i < hg.number_of_edges; i++) {
//         cout << hg.in_neighbors[i] << " ";
//     }
//     cout << endl << "Twohop Offsets:" << endl;
//     for (uint64_t i = 0; i <= hg.number_of_vertices; i++) {
//         cout << hg.twohop_offsets[i] << " ";
//     }
//     cout << endl << "Twohop Neighbors:" << endl;
//     for (uint64_t i = 0; i < hg.number_of_lvl2adj; i++) {
//         cout << hg.twohop_neighbors[i] << " ";
//     }
//     cout << endl << endl;
// }

// void print_GPU_Graph(GPU_Data& h_dd, CPU_Graph& hg)
// {
//     int* number_of_vertices = new int;
//     int* number_of_edges = new int;

//     int* onehop_neighbors = new int[hg.number_of_edges * 2];
//     uint64_t * onehop_offsets = new uint64_t[(hg.number_of_vertices)+1];
//     int* twohop_neighbors = new int[hg.number_of_lvl2adj];
//     uint64_t * twohop_offsets = new uint64_t[(hg.number_of_vertices)+1];

//     chkerr(cudaMemcpy(number_of_vertices, h_dd.number_of_vertices, sizeof(int), cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(number_of_edges, h_dd.number_of_edges, sizeof(int), cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(onehop_neighbors, h_dd.onehop_neighbors, sizeof(int)*hg.number_of_edges * 2, cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(onehop_offsets, h_dd.onehop_offsets, sizeof(uint64_t)*(hg.number_of_vertices+1), cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(twohop_neighbors, h_dd.twohop_neighbors, sizeof(int)*hg.number_of_lvl2adj, cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(twohop_offsets, h_dd.twohop_offsets, sizeof(uint64_t)*(hg.number_of_vertices+1), cudaMemcpyDeviceToHost));

//     cout << endl << " --- (GPU_Graph)device_graph details --- " << endl;
//     cout << endl << "|V|: " << (*number_of_vertices) << " |E|: " << (*number_of_edges) << endl;
//     cout << endl << "Onehop Offsets:" << endl;
//     for (uint64_t i = 0; i <= (*number_of_vertices); i++) {
//         cout << onehop_offsets[i] << " ";
//     }
//     cout << endl << "Onehop Neighbors:" << endl;
//     for (uint64_t i = 0; i < hg.number_of_edges * 2; i++) {
//         cout << onehop_neighbors[i] << " ";
//     }
//     cout << endl << "Twohop Offsets:" << endl;
//     for (uint64_t i = 0; i <= (*number_of_vertices); i++) {
//         cout << twohop_offsets[i] << " ";
//     }
//     cout << endl << "Twohop Neighbors:" << endl;
//     for (uint64_t i = 0; i < hg.number_of_lvl2adj; i++) {
//         cout << twohop_neighbors[i] << " ";
//     }
//     cout << endl << endl;

//     delete number_of_vertices;
//     delete number_of_edges;

//     delete onehop_offsets;
//     delete onehop_neighbors;
//     delete twohop_offsets;
//     delete twohop_neighbors;
// }

void print_CPU_Data(CPU_Data& hd)
{
    cout << endl << " --- (CPU_Data)host_data details --- " << endl;
    cout << endl << "Tasks1: " << "Size: " << (*(hd.tasks1_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(hd.tasks1_count)); i++) {
        cout << hd.tasks1_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].vertexid << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].label << " ";
    }
    cout << endl << "Out-Mem-Deg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].out_mem_deg << " ";
    }
    cout << endl << "Out-Can-Deg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].out_can_deg << " ";
    }
    cout << endl << "In-Mem-Deg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].in_mem_deg << " ";
    }
    cout << endl << "In-Can-Deg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].in_can_deg << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].lvl2adj << " ";
    }

    cout << endl << endl << "Tasks2: " << "Size: " << (*(hd.tasks2_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(hd.tasks2_count)); i++) {
        cout << hd.tasks2_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].vertexid << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].label << " ";
    }
    cout << endl << "Out-Mem-Deg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].out_mem_deg << " ";
    }
    cout << endl << "Out-Can-Deg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].out_can_deg << " ";
    }
    cout << endl << "In-Mem-Deg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].in_mem_deg << " ";
    }
    cout << endl << "In-Can-Deg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].in_can_deg << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].lvl2adj << " ";
    }

    cout << endl << endl << "Buffer: " << "Size: " << (*(hd.buffer_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(hd.buffer_count)); i++) {
        cout << hd.buffer_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].vertexid << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].label << " ";
    }
    cout << endl << "Out-Mem-Deg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].out_mem_deg << " ";
    }
    cout << endl << "Out-Can-Deg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].out_can_deg << " ";
    }
    cout << endl << "In-Mem-Deg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].in_mem_deg << " ";
    }
    cout << endl << "In-Can-Deg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].in_can_deg << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].lvl2adj << " ";
    }
    cout << endl << endl;
}

// void print_GPU_Data(GPU_Data& h_dd, DS_Sizes& dss)
// {
//     uint64_t* current_level = new uint64_t;

//     uint64_t* tasks1_count = new uint64_t;
//     uint64_t* tasks1_offset = new uint64_t[dss.EXPAND_THRESHOLD + 1];
//     Vertex* tasks1_vertices = new Vertex[dss.TASKS_SIZE];

//     uint64_t* buffer_count = new uint64_t;
//     uint64_t* buffer_offset = new uint64_t[dss.BUFFER_OFFSET_SIZE];
//     Vertex* buffer_vertices = new Vertex[dss.BUFFER_SIZE];


//     chkerr(cudaMemcpy(current_level, h_dd.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));

//     chkerr(cudaMemcpy(tasks1_count, h_dd.tasks_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(tasks1_offset, h_dd.tasks_offset, (dss.EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(tasks1_vertices, h_dd.tasks_vertices, (dss.TASKS_SIZE) * sizeof(Vertex), cudaMemcpyDeviceToHost));

//     chkerr(cudaMemcpy(buffer_count, h_dd.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(buffer_offset, h_dd.buffer_offset, (dss.BUFFER_OFFSET_SIZE) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(buffer_vertices, h_dd.buffer_vertices, (dss.BUFFER_SIZE) * sizeof(Vertex), cudaMemcpyDeviceToHost));

//     cout << " --- (GPU_Data)device_data details --- " << endl;
//     cout << endl << "Tasks1: Level: " << (*current_level) << " Size: " << (*tasks1_count) << endl;
//     cout << endl << "Offsets:" << endl;
//     for (int i = 0; i <= (*tasks1_count); i++) {
//         cout << tasks1_offset[i] << " " << flush;
//     }
//     cout << endl << "Vertex:" << endl;
//     for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
//         cout << tasks1_vertices[i].vertexid << " " << flush;
//     }
//     cout << endl << "Label:" << endl;
//     for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
//         cout << tasks1_vertices[i].label << " " << flush;
//     }
//     cout << endl << "Indeg:" << endl;
//     for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
//         cout << tasks1_vertices[i].indeg << " " << flush;
//     }
//     cout << endl << "Exdeg:" << endl;
//     for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
//         cout << tasks1_vertices[i].exdeg << " " << flush;
//     }
//     cout << endl << "Lvl2adj:" << endl;
//     for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
//         cout << tasks1_vertices[i].lvl2adj << " " << flush;
//     }
//     cout << endl;

//     cout << endl << "Buffer: " << "Size: " << (*buffer_count) << endl;
//     cout << endl << "Offsets:" << endl;
//     for (int i = 0; i <= (*buffer_count); i++) {
//         cout << buffer_offset[i] << " " << flush;
//     }
//     cout << endl << "Vertex:" << endl;
//     for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
//         cout << buffer_vertices[i].vertexid << " " << flush;
//     }
//     cout << endl << "Label:" << endl;
//     for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
//         cout << buffer_vertices[i].label << " " << flush;
//     }
//     cout << endl << "Indeg:" << endl;
//     for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
//         cout << buffer_vertices[i].indeg << " " << flush;
//     }
//     cout << endl << "Exdeg:" << endl;
//     for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
//         cout << buffer_vertices[i].exdeg << " " << flush;
//     }
//     cout << endl << "Lvl2adj:" << endl;
//     for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
//         cout << buffer_vertices[i].lvl2adj << " " << flush;
//     }
//     cout << endl;

//     delete current_level;

//     delete tasks1_count;
//     delete tasks1_offset;
//     delete tasks1_vertices;

//     delete buffer_count;
//     delete buffer_offset;
//     delete buffer_vertices;
// }

// void print_All_Warp_Data_Sizes(GPU_Data& h_dd, DS_Sizes& dss)
// {
//     uint64_t* tasks_counts = new uint64_t[NUMBER_OF_WARPS];
//     uint64_t* TASKS_SIZEs = new uint64_t[NUMBER_OF_WARPS];
//     uint64_t* cliques_counts = new uint64_t[NUMBER_OF_WARPS];
//     uint64_t* CLIQUES_SIZEs = new uint64_t[NUMBER_OF_WARPS];

//     chkerr(cudaMemcpy(tasks_counts, h_dd.wtasks_count, sizeof(uint64_t) * NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(cliques_counts, h_dd.wcliques_count, sizeof(uint64_t) * NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
//     for (int i = 0; i < NUMBER_OF_WARPS; i++) {
//         chkerr(cudaMemcpy(TASKS_SIZEs + i, h_dd.wtasks_offset + (i * dss.WTASKS_OFFSET_SIZE) + tasks_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
//         chkerr(cudaMemcpy(CLIQUES_SIZEs + i, h_dd.wcliques_offset + (i * dss.WCLIQUES_OFFSET_SIZE) + cliques_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
//     }

//     cout << "WTasks Sizes: " << flush;
//     for (int i = 0; i < NUMBER_OF_WARPS; i++) {
//         cout << i << ":" << tasks_counts[i] << " " << TASKS_SIZEs[i] << " " << flush;
//     }
//     cout << "\nWCliques Sizez: " << flush;
//     for (int i = 0; i < NUMBER_OF_WARPS; i++) {
//         cout << i << ":" << cliques_counts[i] << " " << CLIQUES_SIZEs[i] << " " << flush;
//     }

//     delete tasks_counts;
//     delete TASKS_SIZEs;
//     delete cliques_counts;
//     delete CLIQUES_SIZEs;
// }

// bool print_Warp_Data_Sizes_Every(GPU_Data& h_dd, int every, DS_Sizes& dss)
// {
//     bool result = false;
//     uint64_t level;
//     chkerr(cudaMemcpy(&level, h_dd.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));
//     if (level % every == 0) {
//         result = print_Warp_Data_Sizes(h_dd, dss);
//     }
//     return result;
// }

// void print_All_Warp_Data_Sizes_Every(GPU_Data& h_dd, int every, DS_Sizes& dss)
// {
//     int level;
//     chkerr(cudaMemcpy(&level, h_dd.current_level, sizeof(int), cudaMemcpyDeviceToHost));
//     if (level % every == 0) {
//         print_All_Warp_Data_Sizes(h_dd, dss);
//     }
// }

// bool print_Data_Sizes_Every(GPU_Data& h_dd, int every, DS_Sizes& dss)
// {
//     bool result = false;
//     int level;
//     chkerr(cudaMemcpy(&level, h_dd.current_level, sizeof(int), cudaMemcpyDeviceToHost));
//     if (level % every == 0) {
//         result = print_Data_Sizes(h_dd, dss);
//     }
//     return result;
// }

// void print_WTask_Buffers(GPU_Data& h_dd, DS_Sizes& dss)
// {
//     uint64_t* wtasks_count = new uint64_t[NUMBER_OF_WARPS];
//     uint64_t* wtasks_offset = new uint64_t[NUMBER_OF_WARPS*dss.WTASKS_OFFSET_SIZE];
//     Vertex* wtasks_vertices = new Vertex[NUMBER_OF_WARPS*dss.WTASKS_SIZE];

//     chkerr(cudaMemcpy(wtasks_count, h_dd.wtasks_count, sizeof(uint64_t)*NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(wtasks_offset, h_dd.wtasks_offset, sizeof(uint64_t) * (NUMBER_OF_WARPS*dss.WTASKS_OFFSET_SIZE), cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(wtasks_vertices, h_dd.wtasks_vertices, sizeof(Vertex) * (NUMBER_OF_WARPS*dss.WTASKS_SIZE), cudaMemcpyDeviceToHost));

//     cout << endl << " --- Warp Task Buffers details --- " << endl;
//     for (int i = 0; i < NUMBER_OF_WARPS; i++) {
//         int wtasks_offset_start = dss.WTASKS_OFFSET_SIZE * i;
//         int wtasks_start = dss.WTASKS_SIZE * i;

//         cout << endl << "Warp " << i << ": " << "Size : " << wtasks_count[i] << endl;
//         if (wtasks_count[i] == 0) {
//             continue;
//         }
//         cout << "Offsets:" << endl;
//         for (int j = 0; j <= wtasks_count[i]; j++) {
//             cout << wtasks_offset[wtasks_offset_start+j] << " ";
//         }
//         cout << endl << "Vertex:" << endl;
//         for (int j = 0; j < wtasks_offset[wtasks_offset_start+wtasks_count[i]]; j++) {
//             cout << wtasks_vertices[wtasks_start+j].vertexid << " ";
//         }
//         cout << endl << "Label:" << endl;
//         for (int j = 0; j < wtasks_offset[wtasks_offset_start + wtasks_count[i]]; j++) {
//             cout << wtasks_vertices[wtasks_start + j].label << " ";
//         }
//         cout << endl << "Indeg:" << endl;
//         for (int j = 0; j < wtasks_offset[wtasks_offset_start + wtasks_count[i]]; j++) {
//             cout << wtasks_vertices[wtasks_start + j].indeg << " ";
//         }
//         cout << endl << "Exdeg:" << endl;
//         for (int j = 0; j < wtasks_offset[wtasks_offset_start + wtasks_count[i]]; j++) {
//             cout << wtasks_vertices[wtasks_start + j].exdeg << " ";
//         }
//         cout << endl << "Lvl2adj:" << endl;
//         for (int j = 0; j < wtasks_offset[wtasks_offset_start + wtasks_count[i]]; j++) {
//             cout << wtasks_vertices[wtasks_start + j].lvl2adj << " ";
//         }
//         cout << endl;
//     }
//     cout << endl << endl;

//     delete wtasks_count;
//     delete wtasks_offset;
//     delete wtasks_vertices;
// }

// void print_WClique_Buffers(GPU_Data& h_dd, DS_Sizes& dss)
// {
//     uint64_t* wcliques_count = new uint64_t[NUMBER_OF_WARPS];
//     uint64_t* wcliques_offset = new uint64_t[NUMBER_OF_WARPS * dss.WCLIQUES_OFFSET_SIZE];
//     int* wcliques_vertex = new int[NUMBER_OF_WARPS * dss.WCLIQUES_SIZE];

//     chkerr(cudaMemcpy(wcliques_count, h_dd.wcliques_count, sizeof(uint64_t) * NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(wcliques_offset, h_dd.wcliques_offset, sizeof(uint64_t) * (NUMBER_OF_WARPS * dss.WTASKS_OFFSET_SIZE), cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(wcliques_vertex, h_dd.wcliques_vertex, sizeof(int) * (NUMBER_OF_WARPS * dss.WTASKS_SIZE), cudaMemcpyDeviceToHost));

//     cout << endl << " --- Warp Clique Buffers details --- " << endl;
//     for (int i = 0; i < NUMBER_OF_WARPS; i++) {
//         int wcliques_offset_start = dss.WTASKS_OFFSET_SIZE * i;
//         int wcliques_start = dss.WTASKS_SIZE * i;

//         cout << endl << "Warp " << i << ": " << "Size : " << wcliques_count[i] << endl;
//         cout << "Offsets:" << endl;
//         for (int j = 0; j <= wcliques_count[i]; j++) {
//             cout << wcliques_offset[wcliques_offset_start + j] << " ";
//         }
//         cout << endl << "Vertex:" << endl;
//         for (int j = 0; j < wcliques_offset[wcliques_offset_start + wcliques_count[i]]; j++) {
//             cout << wcliques_vertex[wcliques_start + j] << " ";
//         }
//     }
//     cout << endl << endl;

//     delete wcliques_count;
//     delete wcliques_offset;
//     delete wcliques_vertex;
// }

// void print_GPU_Cliques(GPU_Data& h_dd, DS_Sizes& dss)
// {
//     uint64_t* cliques_count = new uint64_t;
//     uint64_t* cliques_offset = new uint64_t[dss.CLIQUES_OFFSET_SIZE];
//     int* cliques_vertex = new int[dss.CLIQUES_SIZE];

//     chkerr(cudaMemcpy(cliques_count, h_dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(cliques_offset, h_dd.cliques_offset, sizeof(uint64_t) * dss.CLIQUES_OFFSET_SIZE, cudaMemcpyDeviceToHost));
//     chkerr(cudaMemcpy(cliques_vertex, h_dd.cliques_vertex, sizeof(int) * dss.CLIQUES_SIZE, cudaMemcpyDeviceToHost));

//     cout << endl << " --- (GPU_Cliques)device_cliques details --- " << endl;
//     cout << endl << "Cliques: " << "Size: " << (*cliques_count) << endl;
//     cout << endl << "Offsets:" << endl;
//     for (uint64_t i = 0; i <= (*cliques_count); i++) {
//         cout << cliques_offset[i] << " ";
//     }

//     cout << endl << "Vertex:" << endl;
//     for (uint64_t i = 0; i < (*cliques_count); i++) {
//         cout << i << " S: " << cliques_offset[i] << " E: " << cliques_offset[i+1] << " " << flush;
//         for (uint64_t j = cliques_offset[i]; j < cliques_offset[i + 1]; j++) {
//             cout << cliques_vertex[j] << " " << flush;
//         }
//         cout << endl;
//     }

//     delete cliques_count;
//     delete cliques_offset;
//     delete cliques_vertex;

//     return;

//     cout << endl << "Vertex:" << endl;
//     for (uint64_t i = 0; i < cliques_offset[(*cliques_count)]; i++) {
//         cout << cliques_vertex[i] << " ";
//     }
//     cout << endl;
// }

// void print_CPU_Cliques(CPU_Cliques& hc)
// {
//     cout << endl << " --- (CPU_Cliques)host_cliques details --- " << endl;
//     cout << endl << "Cliques: " << "Size: " << (*(hc.cliques_count)) << endl;
//     cout << endl << "Offsets:" << endl;
//     for (uint64_t i = 0; i <= (*(hc.cliques_count)); i++) {
//         cout << hc.cliques_offset[i] << " ";
//     }
//     cout << endl << "Vertex:" << endl;
//     for (uint64_t i = 0; i < hc.cliques_offset[(*(hc.cliques_count))]; i++) {
//         cout << hc.cliques_vertex[i] << " ";
//     }
//     cout << endl;
// }

void print_vertices(Vertex* vertices, int size)
{
    cout << " --- level 0 details --- " << endl;
    cout << endl << "Tasks1: Level: " << 0 << " Size: " << size << endl;
    cout << endl << "Offsets:" << endl;
    cout << "0 " << size << flush;
    cout << endl << "Vertex:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].vertexid << " " << flush;
    }
    cout << endl << "Label:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].label << " " << flush;
    }
    cout << endl << "Out-Mem-Deg:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].out_mem_deg << " " << flush;
    }
    cout << endl << "Out-Can_deg:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].out_can_deg << " " << flush;
    }
    cout << endl << "In-Mem-Deg:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].in_mem_deg << " " << flush;
    }
    cout << endl << "In-Can-Deg:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].in_can_deg << " " << flush;
    }
    cout << endl << "Lvl2adj:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].lvl2adj << " " << flush;
    }
    cout << endl;
}