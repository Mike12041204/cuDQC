#ifndef DCUQC_HOST_DEBUG_H
#define DCUQC_HOST_DEBUG_H

#include "./common.h"

void print_CPU_Data(CPU_Data& hd);
void print_GPU_Data(GPU_Data& h_dd);
void print_CPU_Graph(CPU_Graph& hg);
void print_GPU_Graph(GPU_Data& h_dd, CPU_Graph& hg);
void print_WTask_Buffers(GPU_Data& h_dd);
void print_WClique_Buffers(GPU_Data& h_dd);
void print_GPU_Cliques(GPU_Data& h_dd); 
void print_CPU_Cliques(CPU_Cliques& hc);
bool print_Data_Sizes(GPU_Data& h_dd, ofstream& output_file);
void h_print_Data_Sizes(CPU_Data& hd, CPU_Cliques& hc, ofstream& output_file);
void print_vertices(Vertex* vertices, int size);
bool print_Data_Sizes_Every(GPU_Data& h_dd, int every, ofstream& output_file);
bool print_Warp_Data_Sizes(GPU_Data& h_dd, ofstream& output_file);
void print_All_Warp_Data_Sizes(GPU_Data& h_dd);
bool print_Warp_Data_Sizes_Every(GPU_Data& h_dd, int every, ofstream& output_file);
void print_All_Warp_Data_Sizes_Every(GPU_Data& h_dd, int every);
void initialize_maxes();
void print_maxes(ofstream& output_file);

#endif // DCUQC_HOST_DEBUG_H