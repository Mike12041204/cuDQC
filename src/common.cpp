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

DS_Sizes::DS_Sizes(const string& filename)
{
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file");
    }

    string line;
    int line_count = 0;
    
    while (getline(file, line)) {
        size_t commaPos = line.find(',');
        if (commaPos != string::npos) {
            string valueStr = line.substr(commaPos + 1);
            uint64_t value = stoull(valueStr);

            switch (line_count) {
                case 0: TASKS_SIZE = value; break;
                case 1: TASKS_PER_WARP = value; break;
                case 2: BUFFER_SIZE = value; break;
                case 3: BUFFER_OFFSET_SIZE = value; break;
                case 4: CLIQUES_SIZE = value; break;
                case 5: CLIQUES_OFFSET_SIZE = value; break;
                case 6: CLIQUES_PERCENT = value; break;
                case 7: WCLIQUES_SIZE = value; break;
                case 8: WCLIQUES_OFFSET_SIZE = value; break;
                case 9: WTASKS_SIZE = value; break;
                case 10: WTASKS_OFFSET_SIZE = value; break;
                case 11: WVERTICES_SIZE = value; break;
            }
            line_count++;
        }
    }

    file.close();

    EXPAND_THRESHOLD = (TASKS_PER_WARP * NUMBER_OF_WARPS);
    CLIQUES_DUMP = (CLIQUES_SIZE * (CLIQUES_PERCENT / 100.0));
}

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