#include "../inc/common.hpp"

// DEBUG - MAX TRACKER VARIABLES
uint64_t mts, mbs, mbo, mcs, mco, wts, wto, wcs, wco, mvs;
ofstream output_file;

// MPI VARIABLES
int wsize;
int grank;
// for every task there is a seperate message buffer and incoming/outgoing handle slot
char msg_buffer[NUMBER_OF_PROCESSESS][100];
// array of handles for messages with all other thread, allows for asynchronous messaging, 
// handles say whether message is complete
MPI_Request rq_send_msg[NUMBER_OF_PROCESSESS];
MPI_Request rq_recv_msg[NUMBER_OF_PROCESSESS];
bool global_free_list[NUMBER_OF_PROCESSESS];

CPU_Graph::CPU_Graph(ifstream& graph_stream)
{
    string line;
    istringstream line_stream;
    int vertex;
    vector<int>* out_nei;
    vector<int>* in_nei;
    int current_line;
    int out_size;
    int in_size;
    uint64_t write_start;

    number_of_vertices = 0;
    while (getline(graph_stream, line)) {
        number_of_vertices++;
    }
    
    out_nei = new vector<int>[number_of_vertices];
    in_nei = new vector<int>[number_of_vertices];
    out_offsets = new uint64_t[number_of_vertices + 1];
    in_offsets = new uint64_t[number_of_vertices + 1];
	twohop_offsets = new uint64_t[number_of_vertices + 1];

	out_offsets[0] = 0;
	in_offsets[0] = 0;

    // reset infile
    graph_stream.clear();
    graph_stream.seekg(0);

    // read all 1hop adj
    current_line = 0;
    number_of_edges = 0;
    while (getline(graph_stream, line)) {
        line_stream.clear();
        line_stream.str(line);

        while (line_stream >> vertex) {
            out_nei[current_line].push_back(vertex);
            in_nei[vertex].push_back(current_line);
        }
        current_line++;
    }

	// sort vectors in ascending manner
	for(int i = 0; i < number_of_vertices; i++){
		sort(out_nei[i].begin(), out_nei[i].end());
		sort(in_nei[i].begin(), in_nei[i].end());

		// NOTE - enforce simple graph, no multi-edges
        auto last = std::unique(out_nei[i].begin(), out_nei[i].end());
        out_nei[i].erase(last, out_nei[i].end());
		last = std::unique(in_nei[i].begin(), in_nei[i].end());
        in_nei[i].erase(last, in_nei[i].end());

		// NOTE - enfore simple graph, no self edges
        out_nei[i].erase(remove(out_nei[i].begin(), out_nei[i].end(), i), out_nei[i].end());
		in_nei[i].erase(remove(in_nei[i].begin(), in_nei[i].end(), i), in_nei[i].end());

		sort(out_nei[i].begin(), out_nei[i].end());
		sort(in_nei[i].begin(), in_nei[i].end());

		// NOTE - count is here as edge adjustment needs to be handled first
        number_of_edges += out_nei[i].size();
	}

	// write to CSR arrays
    out_neighbors = new int[number_of_edges];
    in_neighbors = new int[number_of_edges];

    for(int i = 0; i < number_of_vertices; i++){
        out_size = out_nei[i].size();
        in_size = in_nei[i].size();

        write_start = out_offsets[i];
        out_offsets[i + 1] = write_start + out_size;

		#pragma omp parallel for schedule(static) num_threads(NUMBER_OF_HTHREADS)
        for(int j = 0; j < out_size; j++){
            out_neighbors[write_start + j] = out_nei[i].at(j);
        }

        write_start = in_offsets[i];
        in_offsets[i + 1] = write_start + in_size;

		#pragma omp parallel for schedule(static) num_threads(NUMBER_OF_HTHREADS)
        for(int j = 0; j < in_size; j++){
            in_neighbors[write_start + j] = in_nei[i].at(j);
        }
    }

    GenLevel2NBs();

	delete[] out_nei;
	delete[] in_nei;
}

// create 2-hop neighbors
void CPU_Graph::GenLevel2NBs()
{
	int* temp_int_array1;
	int* temp_int_array2;
	int* temp_int_array3;
	int* temp_int_array4;
	bool* temp_bool_array1;
	vector<int> temp_int_vector1;
	vector<int> temp_int_vector2;
	vector<int> temp_int_vector3;
	vector<int> temp_int_vector4;
	vector<int> temp_int_vector5;
	vector<int>* lvl2adj;
	uint64_t out_size;
	uint64_t in_size;
	uint64_t start_out_size;
	uint64_t start_in_size;
	uint64_t size;
	uint64_t size2;
	uint64_t start_write;
	int vertexid1;
	int vertexid2;
	int round;

	temp_int_array1 = new int[number_of_vertices];
	temp_int_array2 = new int[number_of_vertices];
	temp_int_array3 = new int[number_of_vertices];
	temp_int_array4 = new int[number_of_vertices];
	temp_bool_array1 = new bool[number_of_vertices];
	lvl2adj = new vector<int>[number_of_vertices];

	memset(temp_int_array1, 0, number_of_vertices * sizeof(int));
	memset(temp_int_array2, 0, number_of_vertices * sizeof(int));
	memset(temp_int_array3, 0, number_of_vertices * sizeof(int));
	memset(temp_int_array4, 0, number_of_vertices * sizeof(int));
	memset(temp_bool_array1, 0, number_of_vertices * sizeof(bool));

	twohop_offsets[0] = 0;
	number_of_lvl2adj = 0;

	for(int i = 0; i < number_of_vertices; i++){

		out_size = out_offsets[i + 1] - out_offsets[i];
		in_size = in_offsets[i + 1] - in_offsets[i];

		if(out_size <= 0 || in_size <= 0){
			continue;
		}

		// initialize in and out adjacency DIAs
		start_write = out_offsets[i];
		#pragma omp parallel for schedule(static) num_threads(NUMBER_OF_HTHREADS)
		for(uint64_t j = 0; j < out_size; j++){
			temp_int_array1[out_neighbors[start_write + j]] = 1;
		}
		start_write = in_offsets[i];
		#pragma omp parallel for schedule(static) num_threads(NUMBER_OF_HTHREADS)
		for(uint64_t j = 0; j < in_size; j++){
			temp_int_array2[in_neighbors[start_write + j]] = 1;
		}

		// find all bi-neighbors
		start_write = out_offsets[i];
		for(uint64_t j = 0; j < out_size; j++){
			vertexid1 = out_neighbors[start_write + j];

			if(temp_int_array2[vertexid1] == 1){
				temp_int_vector5.push_back(vertexid1);
			}
			else{
				temp_int_array3[vertexid1] = 1;
				temp_int_vector1.push_back(vertexid1);
			}
		}

		start_write = in_offsets[i];
		for(uint64_t j = 0; j < in_size; j++){

			vertexid1 = in_neighbors[start_write + j];

			if(temp_int_array1[vertexid1] == 0){
				temp_int_array4[vertexid1] = 1;
				temp_int_vector2.push_back(vertexid1);
			}
		}

		round = 1;
		do{
			start_out_size = temp_int_vector1.size();
			start_in_size = temp_int_vector2.size();

			// add bi-neighbors to in and out neighbors
			size = temp_int_vector5.size();
			for(uint64_t j = 0; j < size; j++){
				vertexid1 = temp_int_vector5.at(j);

				temp_int_vector1.push_back(j);
				temp_int_vector2.push_back(j);
			}

			// perform pruning
			size = temp_int_vector1.size();
			for(uint64_t j = 0; j < size; j++){

				vertexid1 = temp_int_vector1.at(j);
				size2 = out_offsets[vertexid1 + 1] - out_offsets[vertexid1];
				start_write = out_offsets[vertexid1];

				for(uint64_t k = 0; k < size2; k++){
					vertexid2 = out_neighbors[start_write + k];

					if(temp_int_array4[vertexid2] == round){
						temp_int_array4[vertexid2]++;
						temp_int_vector4.push_back(vertexid2);
					}
				}
			}

			size = temp_int_vector2.size();
			for(uint64_t j = 0; j < size; j++){

				vertexid1 = temp_int_vector2.at(j);
				size2 = in_offsets[vertexid1 + 1] - in_offsets[vertexid1];
				start_write = in_offsets[vertexid1];

				for(uint64_t k = 0; k < size2; k++){
					vertexid2 = in_neighbors[start_write + k];

					if(temp_int_array3[vertexid2] == round){
						temp_int_array3[vertexid2]++;
						temp_int_vector3.push_back(vertexid2);
					}
				}
			}

			// reset temporary vectors
			temp_int_vector1.swap(temp_int_vector3);
			temp_int_vector2.swap(temp_int_vector4);

			temp_int_vector3.clear();
			temp_int_vector4.clear();

			// go to next round
			round++;

		}while(temp_int_vector1.size() < start_out_size || temp_int_vector2.size() < 
			   start_in_size);
	
		// reset temp arrays
		start_write = out_offsets[i];
		#pragma omp parallel for schedule(static) num_threads(NUMBER_OF_HTHREADS)
		for(uint64_t j = 0; j < out_size; j++){
			temp_int_array1[out_neighbors[start_write + j]] = 0;
			temp_int_array3[out_neighbors[start_write + j]] = 0;
		}
		start_write = in_offsets[i];
		#pragma omp parallel for schedule(static) num_threads(NUMBER_OF_HTHREADS)
		for(uint64_t j = 0; j < in_size; j++){
			temp_int_array2[in_neighbors[start_write + j]] = 0;
			temp_int_array4[in_neighbors[start_write + j]] = 0;
		}

		// add bi-neighbors, O, and I twohop adj into results
		size = temp_int_vector5.size();
		for(uint64_t j = 0; j < size; j++){
			vertexid1 = temp_int_vector5.at(j);
			lvl2adj[i].push_back(vertexid1);
			temp_bool_array1[vertexid1] = true;
		}
		size = temp_int_vector1.size();
		for(uint64_t j = 0; j < size; j++){
			vertexid1 = temp_int_vector1.at(j);
			lvl2adj[i].push_back(vertexid1);
			temp_bool_array1[vertexid1] = true;
		}
		size = temp_int_vector2.size();
		for(uint64_t j = 0; j < size; j++){
			vertexid1 = temp_int_vector2.at(j);
			lvl2adj[i].push_back(vertexid1);
			temp_bool_array1[vertexid1] = true;
		}

		// procees B twohop adj, adj with no direct connections
		size = temp_int_vector5.size();
		for(uint64_t j = 0; j < size; j++){
			vertexid1 = temp_int_vector5.at(j);
			temp_int_vector1.push_back(vertexid1);
			temp_int_vector2.push_back(vertexid1);
		}

		// perform unions described in paper to find B
		size = temp_int_vector1.size();
		for(uint64_t j = 0; j < size; j++){

			vertexid1 = temp_int_vector1.at(j);
			size2 = out_offsets[vertexid1 + 1] - out_offsets[vertexid1];
			start_write = out_offsets[vertexid1];

			for(uint64_t k = 0; k < size2; k++){
				vertexid2 = out_neighbors[start_write + k];

				if(vertexid2 != i && temp_bool_array1[vertexid2] == false && 
				   temp_int_array1[vertexid2] != 1){

				    temp_int_vector3.push_back(vertexid2);
					temp_int_array1[vertexid2] = 1;
				}
			}
		}

		size = temp_int_vector1.size();
		for(uint64_t j = 0; j < size; j++){

			vertexid1 = temp_int_vector1.at(j);
			size2 = in_offsets[vertexid1 + 1] - in_offsets[vertexid1];
			start_write = in_offsets[vertexid1];

			#pragma omp parallel for schedule(static) num_threads(NUMBER_OF_HTHREADS)
			for(uint64_t k = 0; k < size2; k++){
				vertexid2 = in_neighbors[start_write + k];

				if(temp_int_array1[vertexid2] == 1){
					temp_int_array1[vertexid2] = 2;
				}
			}
		}

		size = temp_int_vector2.size();
		for(uint64_t j = 0; j < size; j++){

			vertexid1 = temp_int_vector2.at(j);
			size2 = out_offsets[vertexid1 + 1] - out_offsets[vertexid1];
			start_write = out_offsets[vertexid1];

			#pragma omp parallel for schedule(static) num_threads(NUMBER_OF_HTHREADS)
			for(uint64_t k = 0; k < size2; k++){
				vertexid2 = out_neighbors[start_write + k];

				if(temp_int_array1[vertexid2] == 2){
					temp_int_array1[vertexid2] = 3;
				}
			}
		}

		size = temp_int_vector2.size();
		for(uint64_t j = 0; j < size; j++){

			vertexid1 = temp_int_vector2.at(j);
			size2 = in_offsets[vertexid1 + 1] - in_offsets[vertexid1];
			start_write = in_offsets[vertexid1];

			for(uint64_t k = 0; k < size2; k++){
				vertexid2 = in_neighbors[start_write + k];

				if(temp_int_array1[vertexid2] == 3 && temp_bool_array1[vertexid2] == false){
					temp_bool_array1[vertexid2] = true;
					lvl2adj[i].push_back(vertexid2);
				}
			}
		}

		// sort and count this vertices twohop adj
		sort(lvl2adj[i].begin(), lvl2adj[i].end());
		number_of_lvl2adj += lvl2adj[i].size();

		// reset temp arrays
		size = temp_int_vector3.size();
		for(uint64_t j = 0; j < size; j++){
			vertexid1 = temp_int_vector3.at(j);
			temp_int_array1[vertexid1] = 0;
		}
		size = lvl2adj[i].size();
		for(uint64_t j = 0; j < size; j++){
			vertexid1 = lvl2adj[i].at(j);
			temp_bool_array1[vertexid1] = false;
		}
		temp_int_vector1.clear();
		temp_int_vector2.clear();
		temp_int_vector3.clear();
		temp_int_vector5.clear();
	}

	twohop_neighbors = new int[number_of_lvl2adj];

	for(int i = 0; i < number_of_vertices; i++){
		size = lvl2adj[i].size();
		start_write = twohop_offsets[i];
		twohop_offsets[i + 1] = start_write + size;

		for(uint64_t j = 0; j < size; j++){
			vertexid1 = lvl2adj[i].at(j);
			twohop_neighbors[start_write + j] = vertexid1;
		}
	}

	delete[] temp_int_array1;
	delete[] temp_int_array2;
	delete[] temp_int_array3;
	delete[] temp_int_array4;
	delete[] temp_bool_array1;
	delete[] lvl2adj;
}

CPU_Graph::~CPU_Graph() 
{
    delete[] out_offsets;
	delete[] out_neighbors;
	delete[] in_offsets;
	delete[] in_neighbors;
	delete[] twohop_offsets;
	delete[] twohop_neighbors;
	delete[] original_id_map;
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
                case 1: EXPAND_THRESHOLD = value; break;
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
}