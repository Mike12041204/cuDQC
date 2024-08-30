#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <time.h>
#include <chrono>
#include <cstring>
#include <sys/timeb.h>
#include <omp.h>
using namespace std;

// omp settings
#define NUMBER_OF_HTHREADS 2

// CPU GRAPH / CONSTRUCTOR
class CPU_Graph
{
    public:

    int number_of_vertices;
    int number_of_edges;
    uint64_t number_of_lvl2adj;
    // one dimentional arrays of 1hop and 2hop neighbors and the offsets for each vertex
    int* out_neighbors;
    uint64_t* out_offsets;
    int* in_neighbors;
    uint64_t* in_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;

    CPU_Graph(ifstream& graph_stream);
    ~CPU_Graph();
    void write_serialized(string output);
    void GenLevel2NBs();
};

int h_sort_asce(const void* a, const void* b);
void print_CPU_Graph(CPU_Graph& hg);
int comp_int(const void *e1, const void *e2);

// MAIN
int main(int argc, char* argv[])
{
    // ENSURE PROPER USAGE
    if (argc != 6) {
        printf("Usage: ./main <file_path> <graph_file> <out_gamma> <in_gamma> <min_size>\n");
        return 1;
    }

	string adj_path = string(argv[1]) + argv[2];
	string ser_path = "./DQC-S0_" + string(argv[2]) + "-" + argv[3] + "-" + argv[4] + "-" + argv[5];

    ifstream graph_stream(ser_path, ios::in);
    if (graph_stream.is_open()) {
        cout << ">:SERIALIZED GRAPH FOUND" << endl;
		return 0;
    }

	cout << ">:SERIALIZED GRAPH NOT FOUND" << endl;

	graph_stream.open(adj_path, ios::in);
	if(!graph_stream.is_open()){
		cout << ">:INVALID ADJACENCY LIST GRAPH FILE" << endl;
		return 1;
	}
	cout << ">:CREATING SERIALIZED GRAPH" << endl;

    // GRAPH
    CPU_Graph hg(graph_stream);
	cout << ser_path << endl;
    hg.write_serialized(ser_path);
	cout << "!" << endl;
    graph_stream.close();
    
    return 0;
}

// sorts degrees in ascending order
int h_sort_asce(const void* a, const void* b)
{
    int n1;
    int n2;

    n1 = *(int*)a;
    n2 = *(int*)b;

    if (n1 < n2) {
        return -1;
    }
    else if (n1 > n2) {
        return 1;
    }
    else {
        return 0;
    }
}

int comp_int(const void *e1, const void *e2)
{
	int n1, n2;
	n1 = *(int *) e1;
	n2 = *(int *) e2;

	if (n1>n2)
		return 1;
	else if (n1<n2)
		return -1;
	else
		return 0;
}

void print_CPU_Graph(CPU_Graph& hg) {
    cout << endl << " --- (CPU_Graph)host_graph details --- " << endl;
    cout << endl << "|V|: " << hg.number_of_vertices << " |E|: " << hg.number_of_edges << endl;
    cout << endl << "Out Offsets:" << endl;
    for (int i = 0; i <= hg.number_of_vertices; i++) {
        cout << hg.out_offsets[i] << " ";
    }
    cout << endl << "Out Neighbors:" << endl;
    for (int i = 0; i < hg.number_of_edges; i++) {
        cout << hg.out_neighbors[i] << " ";
    }
    cout << endl << "In Offsets:" << endl;
    for (int i = 0; i <= hg.number_of_vertices; i++) {
        cout << hg.in_offsets[i] << " ";
    }
    cout << endl << "In Neighbors:" << endl;
    for (int i = 0; i < hg.number_of_edges; i++) {
        cout << hg.in_neighbors[i] << " ";
    }
    cout << endl << "Twohop Offsets:" << endl;
    for (uint64_t i = 0; i <= hg.number_of_vertices; i++) {
        cout << hg.twohop_offsets[i] << " ";
    }
    cout << endl << "Twohop Neighbors:" << endl;
    for (uint64_t i = 0; i < hg.number_of_lvl2adj; i++) {
        cout << hg.twohop_neighbors[i] << " ";
    }
    cout << endl << endl;
}

CPU_Graph::CPU_Graph(ifstream& graph_stream)
{
    string line;
    istringstream line_stream;
    int vertex;
    vector<int>* out_nei;
    vector<int>* in_nei;
    vector<int>* lvl2_nei;
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
    lvl2_nei = new vector<int>[number_of_vertices];
    out_offsets = new uint64_t[number_of_vertices + 1];
    in_offsets = new uint64_t[number_of_vertices + 1];
	twohop_offsets = new uint64_t[number_of_vertices + 1];

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

            number_of_edges++;
        }
        current_line++;
    }

    out_neighbors = new int[number_of_edges];
    in_neighbors = new int[number_of_edges];

    for(int i = 0; i < number_of_vertices; i++){
        out_size = out_nei[i].size();
        in_size = in_nei[i].size();

        write_start = out_offsets[i];
        out_offsets[i + 1] = write_start + out_size;

        for(int j = 0; j < out_size; j++){
            out_neighbors[write_start + j] = out_nei[i].at(j);
        }

        write_start = in_offsets[i];
        in_offsets[i + 1] = write_start + in_size;

        for(int j = 0; j < in_size; j++){
            in_neighbors[write_start + j] = in_nei[i].at(j);
        }
    }

    GenLevel2NBs();
}

// create 2-hop neighbors
void CPU_Graph::GenLevel2NBs()
{
	// each thread has arrays to work with
	int** set_out_single, **set_in_single, **temp_array, **temp_array2, **pnb_list;
	bool** pbflags;
	set_out_single = new int*[NUMBER_OF_HTHREADS];
	set_in_single = new int*[NUMBER_OF_HTHREADS];
	temp_array = new int*[NUMBER_OF_HTHREADS];
	temp_array2 = new int*[NUMBER_OF_HTHREADS];
	pnb_list = new int*[NUMBER_OF_HTHREADS];
	pbflags = new bool*[NUMBER_OF_HTHREADS];

	//-------
	// initialize each threads arrays
#pragma omp parallel for schedule(dynamic, 1) num_threads(NUMBER_OF_HTHREADS)
	for(int i=0; i<NUMBER_OF_HTHREADS; i++)
	{
		// these are all used as DIA arrays based on vertexids
		set_out_single[i] = new int[number_of_vertices];
		set_in_single[i] = new int[number_of_vertices];
		temp_array[i] = new int[number_of_vertices];
		temp_array2[i] = new int[number_of_vertices];
		pnb_list[i] = new int[number_of_vertices];
		pbflags[i] = new bool[number_of_vertices];

		memset(set_out_single[i], 0, sizeof(int)*number_of_vertices);
		memset(set_in_single[i], 0, sizeof(int)*number_of_vertices);
		memset(temp_array[i], 0, sizeof(int)*number_of_vertices); //out
		memset(temp_array2[i], 0, sizeof(int)*number_of_vertices); //in
		memset(pbflags[i], 0, sizeof(bool)*number_of_vertices);
	}

	//-------
//	int* set_out_single = new int[number_of_vertices];
//	int* set_in_single = new int[number_of_vertices];
//	memset(set_out_single, 0, sizeof(int)*number_of_vertices);
//	memset(set_in_single, 0, sizeof(int)*number_of_vertices);

	// initialize 2hop adj write location
	int** mpplvl2_nbs = new int*[number_of_vertices]; // mpplvl2_nbs[i] = node i's level-2 neighbors, first element keeps the 2-hop-list length

	// initialize vectors for temp work
	vector<int> bi_nbs[NUMBER_OF_HTHREADS];
	vector<int> vec_out[NUMBER_OF_HTHREADS];
	vector<int> vec_in[NUMBER_OF_HTHREADS];
	vector<int> temp_vec_out[NUMBER_OF_HTHREADS];
	vector<int> temp_vec_in[NUMBER_OF_HTHREADS];
	vector<int> temp_vec[NUMBER_OF_HTHREADS];

	// for each vertex
#pragma omp parallel for schedule(dynamic, 1) num_threads(NUMBER_OF_HTHREADS)
	for(int i=0; i<number_of_vertices; i++)
	{
		// temp array is out adj, temp array 2 is in adj

		int tid = omp_get_thread_num();

		// vertices number of out adj
		int out_size = out_offsets[i + 1] - out_offsets[i];
		// vertices number of in adj
		int in_size = in_offsets[i + 1] - in_offsets[i];

		// if the vertex has both in and out adj
		if(out_size > 0 && in_size > 0)
		{
			// set DIA so all vertices which are out adj to vertex i have value 1
			for(int j=1; j<=out_size; j++)
				temp_array[tid][out_neighbors[out_offsets[i] + j - 1]] = 1;

			// same for in adj
			for(int j=1; j<=in_size; j++)
				temp_array2[tid][in_neighbors[in_offsets[i] + j - 1]] = 1;


			//get bidirectional connected neighbors, O and I
			// for all out adj
			for(int j=1; j<=out_size; j++)
			{
				// get the out adj vertexid
				int v = out_neighbors[out_offsets[i] + j - 1];

				// if the out adj is also in adj
				if(temp_array2[tid][v] == 1)

					// add the vertex to bidirectional adj vector
					bi_nbs[tid].push_back(v);

				// else the adj is only out
				else
				{
					// add to DIA and vector for out adj
					// MIKE - do we really need set_out_single and set_in_single? I feel we should just continue to reference temp arrays since the are already DIA
					set_out_single[tid][v] = 1;
					vec_out[tid].push_back(v);
				}
			}

			// for all in adj
			for(int j=1; j<=in_size; j++)
			{
				// get in adj vertex id
				int v = in_neighbors[in_offsets[i] + j - 1];

				// if in adj is NOT also out adj
				if(temp_array[tid][v] == 0)
				{
					// add to DIA and vector for in adj
					set_in_single[tid][v] = 1;
					vec_in[tid].push_back(v);
				}
			}

			// MIKE - at this point:
			// temp_array: DIA for each vertex on out adj
			// temp_array2: DIA for each vertex on in adj
			// bi_nbs: vector of bidirectional adj
			// vec_out: vector of out only adj
			// vec_in: vector of in only adj
			// set_out_single: DIA for each vertex on out only adj
			// set_in_single: DIA for each vertex on in only adj

			// repeat pruning
			int round = 1;
			int nb;
			int out_single_size, in_single_size;
			do {
				//update So and Si
				// size of only out and in vertices
				out_single_size = vec_out[tid].size();
				in_single_size = vec_in[tid].size();

				// for all bi adj
				for(int j=0; j<bi_nbs[tid].size();j++)
				{
					// add back in bi adj to out and in adj
					vec_out[tid].push_back(bi_nbs[tid][j]);
					vec_in[tid].push_back(bi_nbs[tid][j]);
				}

				//check the 1-hop neighbors which only connected in one direction
				// for all out adj
				for(int j=0; j<vec_out[tid].size();j++)
				{
					// get out adj vertexid
					int vn = vec_out[tid][j];

					// for all out adj of out vertex
					for(int k=1; k<=out_offsets[vn + 1] - out_offsets[vn]; k++)
					{
						// get vertexid of new out adj
						nb = out_neighbors[out_offsets[vn] + k - 1];

						// if it is a current valid in adj 
						if(set_in_single[tid][nb] == round)
						{
							// continue it as a valid in adj
							set_in_single[tid][nb]++;

							// add it to the next level in adj vector
							temp_vec_in[tid].push_back(nb);
						}
					}
				}

				// for all in adj, same as last for
				for(int j=0; j<vec_in[tid].size();j++)
				{
					// get vertexid
					int vn = vec_in[tid][j];

					// for all in adj of in adj
					for(int k=1; k<=in_offsets[vn + 1] - in_offsets[vn]; k++)
					{
						// get vertexid
						nb = in_neighbors[in_offsets[vn] + k - 1];

						// if vertex is out adj
						if(set_out_single[tid][nb] == round)
						{
							// continue as out adj
							set_out_single[tid][nb]++;

							// add out adj to next level vector
							temp_vec_out[tid].push_back(nb);
						}
					}
				}

				// set out and in vectors as temp next level ones
				vec_out[tid].swap(temp_vec_out[tid]);
				vec_in[tid].swap(temp_vec_in[tid]);

				// clear temp vectors
				temp_vec_out[tid].clear();
				temp_vec_in[tid].clear();

				// increment round counter
				round++;

				// continue while some in or out adj has been removed in the last level
			} while(vec_out[tid].size()<out_single_size || vec_in[tid].size()<in_single_size);

			//reset single set
			for(int j=1; j<=out_size; j++)
				set_out_single[tid][out_neighbors[out_offsets[i] + j - 1]] = 0;

			for(int j=1; j<=in_size; j++)
				set_in_single[tid][in_neighbors[in_offsets[i] + j - 1]] = 0;
			//reset gptemp_array
			for(int j=1; j<=out_size; j++)
				temp_array[tid][out_neighbors[out_offsets[i] + j - 1]] = 0;

			for(int j=1; j<=in_size; j++)
				temp_array2[tid][in_neighbors[in_offsets[i] + j - 1]] = 0;

			//add bidirectional neighbors into 2hop nbs
			int nlist_len = 0;
			for (int j=0; j<bi_nbs[tid].size(); j++)
			{
				pnb_list[tid][nlist_len++] = bi_nbs[tid][j];
				pbflags[tid][bi_nbs[tid][j]] = true;
			}

			//add single out & in 1hop neighbors
			for (int j=0; j<vec_out[tid].size(); j++)
			{
				pnb_list[tid][nlist_len++] = vec_out[tid][j];
				pbflags[tid][vec_out[tid][j]] = true;
			}
			for (int j=0; j<vec_in[tid].size(); j++)
			{
				pnb_list[tid][nlist_len++] = vec_in[tid][j];
				pbflags[tid][vec_in[tid][j]] = true;
			}

			// MIKE - at this point
			// pnb_list: has the twohop adj from onehop adj, O, and I (needs B)
			// pbflags: is DIA for each vertex on whether it is a twohop adj or not

			// add back bi out and in adj to out and in adj vectors
			for(int j=0; j<bi_nbs[tid].size();j++)
			{
				vec_out[tid].push_back(bi_nbs[tid][j]);
				vec_in[tid].push_back(bi_nbs[tid][j]);
			}

			// MIKE - the next steps are performign the four unions of intersections described in the paper to find case B twohop adj

			// for all out adj
			for (int j=0; j<vec_out[tid].size(); j++)
			{
				// get vertexid
				int u = vec_out[tid][j];

				// for all out adj of out adj
				for(int k=1; k<=out_offsets[u + 1] - out_offsets[u]; k++)
				{
					// get vertexid
					int v = out_neighbors[out_offsets[u] + k - 1];

					// if vertex is not self and is not already 2hop adj and is not already considered in this step
					if(v != i && pbflags[tid][v] == false && temp_array[tid][v] != 1)
					{
						// add to temp vector and mark as considered
						// MIKE - do we need temp_vec here? Don't seem to use it
						temp_vec[tid].push_back(v);
						temp_array[tid][v]=1;
					}
				}
			}

			// for all out adj
			for (int j=0; j<vec_out[tid].size(); j++)
			{
				// get vertexid
				int u = vec_out[tid][j];

				// for all in adj of out adj
				for(int k=1; k<=in_offsets[u + 1] - in_offsets[u]; k++)
				{
					// get vertexid
					int v = in_neighbors[in_offsets[u] + k - 1];

					// if made it through last step
					if(temp_array[tid][v] == 1)

						// mark as passing this step
						temp_array[tid][v]=2;
				}
			}

			// for all in adj
			for (int j=0; j<vec_in[tid].size(); j++)
			{
				// get vertexid
				int u = vec_in[tid][j];

				// for all out out adj of in adj
				for(int k=1; k<=out_offsets[u + 1] - out_offsets[u]; k++)
				{
					// get vertexid
					int v = out_neighbors[out_offsets[u] + k - 1];

					// if vertex passes last step
					if(temp_array[tid][v] == 2)

						// mark as passing this step
						temp_array[tid][v]=3;
				}
			}

			// for all in adj
			for (int j=0; j<vec_in[tid].size(); j++)
			{
				// get vertexid
				int u = vec_in[tid][j];

				// for all in adj of in adj
				for(int k=1; k<=in_offsets[u + 1] - in_offsets[u]; k++)
				{
					// get vertexid
					int v = in_neighbors[in_offsets[u] + k - 1];

					// if vertex passed last step and is not alreayd in twohop adj
					if(temp_array[tid][v] == 3 && pbflags[tid][v] == false)
					{
						// add to twohop adj
						pbflags[tid][v] = true;
						pnb_list[tid][nlist_len++] = v;
					}
				}
			}

			//reset gptemp_array
			int temp_vec_size = temp_vec[tid].size();
			for(int j=0; j<temp_vec_size; j++)
				temp_array[tid][temp_vec[tid][j]] = 0;

			// sort twohop adj
			if(nlist_len>1)
				qsort(pnb_list[tid], nlist_len, sizeof(int), comp_int);
			
			// allocate and copy final twohop adj location
			mpplvl2_nbs[i] = new int[nlist_len+1];
			mpplvl2_nbs[i][0] = nlist_len; //first element keeps the 2-hop-list length
			if(nlist_len>0)
				memcpy(&mpplvl2_nbs[i][1], pnb_list[tid], sizeof(int)*nlist_len);

			// reset flags
			for(int j=0;j<nlist_len;j++)
				pbflags[tid][pnb_list[tid][j]] = false;

			// clear memory
			bi_nbs[tid].clear();
			vec_out[tid].clear();
			vec_in[tid].clear();
			temp_vec[tid].clear();
		} else {
			mpplvl2_nbs[i] = new int[1];
			mpplvl2_nbs[i][0] = 0;
		}
	}

	// transfer twohop offsets
	// TODO - make parallel
	twohop_offsets[0] = 0;
	for(int i = 0; i < number_of_vertices; i++){
		twohop_offsets[i + 1] = twohop_offsets[i] + mpplvl2_nbs[i][0];
	}

	number_of_lvl2adj = twohop_offsets[number_of_vertices];

	twohop_neighbors = new int[number_of_lvl2adj];
	
	// transfer twohop neighbors
	for(int i = 0; i < number_of_vertices; i++){
		for(uint64_t k = 0; k < twohop_offsets[i + 1] - twohop_offsets[i]; k++){
			twohop_neighbors[twohop_offsets[i] + k] = mpplvl2_nbs[i][k + 1];
		}
	}

	for(int i=0; i<NUMBER_OF_HTHREADS; i++)
	{
		delete []pbflags[i];
		delete []pnb_list[i];
		delete []set_out_single[i];
		delete []set_in_single[i];
		delete []temp_array[i];
		delete []temp_array2[i];
	}
	delete []pbflags;
	delete []pnb_list;
	delete []set_out_single;
	delete []set_in_single;
	delete []temp_array;
	delete []temp_array2;
}

void CPU_Graph::write_serialized(string output)
{
	ofstream buffer_file;

    buffer_file.open(output, ios::binary);

    buffer_file.write(reinterpret_cast<const char*>(&number_of_vertices), sizeof(int));
	buffer_file.write(reinterpret_cast<const char*>(&number_of_edges), sizeof(int));
	buffer_file.write(reinterpret_cast<const char*>(&number_of_lvl2adj), sizeof(uint64_t));
    buffer_file.write(reinterpret_cast<const char*>(out_offsets), (number_of_vertices + 1) * sizeof(uint64_t));
    buffer_file.write(reinterpret_cast<const char*>(out_neighbors), number_of_edges * sizeof(int));
	buffer_file.write(reinterpret_cast<const char*>(in_offsets), (number_of_vertices + 1) * sizeof(uint64_t));
    buffer_file.write(reinterpret_cast<const char*>(in_neighbors), number_of_edges * sizeof(int));
	buffer_file.write(reinterpret_cast<const char*>(twohop_offsets), (number_of_vertices + 1) * sizeof(uint64_t));
    buffer_file.write(reinterpret_cast<const char*>(twohop_neighbors), number_of_lvl2adj * sizeof(int));

    buffer_file.close();
}

// TODO - finish this method
CPU_Graph::~CPU_Graph()
{
    // delete[] out_offsets;
	// delete[] out_neighbors;
	// delete[] in_neighbors;
	// delete[] in_neighbors;
	// delete[] twohop_offsets;
	// delete[] twohop_neighbors;
}