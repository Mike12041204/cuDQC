#ifndef DCUQC_HOST_GENERAL_H
#define DCUQC_HOST_GENERAL_H

#include "./common.h"

void calculate_minimum_degrees(CPU_Graph& hg);
void search(CPU_Graph& hg, ofstream& temp_results, ofstream& output_file);
void allocate_memory(CPU_Data& hd, GPU_Data& dd, CPU_Cliques& hc, CPU_Graph& hg);
void initialize_tasks(CPU_Graph& hg, CPU_Data& hd);
void move_to_gpu(CPU_Data& hd, GPU_Data& dd);
void dump_cliques(CPU_Cliques& hc, GPU_Data& dd, ofstream& output_file);
void flush_cliques(CPU_Cliques& hc, ofstream& temp_results);
void free_memory(CPU_Data& hd, GPU_Data& dd, CPU_Cliques& hc);

#endif // DCUQC_HOST_GENERAL_H