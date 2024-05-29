#include "../inc/common.h"

// DEBUG - MAX TRACKER VARIABLES
uint64_t mts, mbs, mbo, mcs, mco, wts, wto, wcs, wco, mvs;

// COMMAND LINE INPUT VARIABLES
double minimum_degree_ratio;
int minimum_clique_size;
int* minimum_degrees;
int scheduling_toggle;

// MPI VARIABLES
int wsize;
int grank;

CPU_Graph::CPU_Graph(ifstream& graph_stream)
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

CPU_Graph::~CPU_Graph() 
{
    delete onehop_neighbors;
    delete onehop_offsets;
    delete twohop_neighbors;
    delete twohop_offsets;
}