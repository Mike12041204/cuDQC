#include "../inc/common.hpp"

// DEBUG - MAX TRACKER VARIABLES
uint64_t mts, mbs, mbo, mcs, mco, wts, wto, wcs, wco, mvs;
ofstream output_file;

// MPI VARIABLES
int wsize;
int grank;
char msg_buffer[NUMBER_OF_PROCESSESS][100];             // for every task there is a seperate message buffer and incoming/outgoing handle slot 
MPI_Request rq_send_msg[NUMBER_OF_PROCESSESS];          // array of handles for messages with all other thread, allows for asynchronous messaging, handles say whether message is complete
MPI_Request rq_recv_msg[NUMBER_OF_PROCESSESS];
bool global_free_list[NUMBER_OF_PROCESSESS];

CPU_Graph::CPU_Graph(ifstream& graph_stream)
{ 
    graph_stream.read(reinterpret_cast<char*>(&number_of_vertices), sizeof(int));
	graph_stream.read(reinterpret_cast<char*>(&number_of_edges), sizeof(int));
	graph_stream.read(reinterpret_cast<char*>(&number_of_lvl2adj), sizeof(uint64_t));

    out_offsets = new uint64_t[number_of_vertices + 1];
    in_offsets = new uint64_t[number_of_vertices + 1];
	twohop_offsets = new uint64_t[number_of_vertices + 1];
    out_neighbors = new int[number_of_edges];
    in_neighbors = new int[number_of_edges];
    twohop_neighbors = new int[number_of_lvl2adj];
    
    graph_stream.read(reinterpret_cast<char*>(out_offsets), (number_of_vertices + 1) * sizeof(uint64_t));
    graph_stream.read(reinterpret_cast<char*>(out_neighbors), number_of_edges * sizeof(int));
	graph_stream.read(reinterpret_cast<char*>(in_offsets), (number_of_vertices + 1) * sizeof(uint64_t));
    graph_stream.read(reinterpret_cast<char*>(in_neighbors), number_of_edges * sizeof(int));
	graph_stream.read(reinterpret_cast<char*>(twohop_offsets), (number_of_vertices + 1) * sizeof(uint64_t));
    graph_stream.read(reinterpret_cast<char*>(twohop_neighbors), number_of_lvl2adj * sizeof(int));
}

CPU_Graph::~CPU_Graph() 
{
    delete[] out_offsets;
	delete[] out_neighbors;
	delete[] in_offsets;
	delete[] in_neighbors;
	delete[] twohop_offsets;
	delete[] twohop_neighbors;
}

DS_Sizes::DS_Sizes(const string& filename)
{
    ifstream file(filename);
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
                case 12: DEBUG_TOGGLE = value; break;
            }
        }

        line_count++;
    }

    file.close();

    EXPAND_THRESHOLD = (TASKS_PER_WARP * NUMBER_OF_WARPS);
    CLIQUES_DUMP = (CLIQUES_OFFSET_SIZE * (CLIQUES_PERCENT / 100.0));
}