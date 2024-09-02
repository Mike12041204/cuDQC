#include "./inc/common.hpp"
//#include "./inc/host_functions.hpp"
//#include "./inc/host_debug.h"

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

// MAIN
int main(int argc, char* argv[])
{
    // TIME
    auto start2 = chrono::high_resolution_clock::now();

    double minimum_out_degree_ratio;
    double minimum_in_degree_ratio;
    int minimum_clique_size;            // minimum size for cliques
    int* minimum_out_degrees;           // stores the minimum connections per vertex for all size cliques
    int* minimum_in_degrees;
    string filename;                    // used in concatenation for making filenames
    ifstream read_file;                 // multiple read files
    ofstream write_file;                // writing results to mutiple files
    string line;                        // stores lines from read file
    string output;

    // ENSURE PROPER USAGE
    if (argc != 7) {
        printf("Usage: ./program1 <graph_file> <out_gamma> <in_gamma> <min_size> <ds_sizes_file> <output_file>\n");
        return 1;
    }
    read_file.open(argv[5], ios::in);
    if(!read_file.is_open()){
        cout << "invalid data structure sizes file\n" << endl;
    }
    read_file.close();
    // reads the sizes of the data structures
    DS_Sizes dss(argv[5]);
    read_file.open(argv[1], ios::binary);
    if (!read_file.is_open()) {
        printf("invalid graph file\n");
        return 1;
    }
    minimum_out_degree_ratio = atof(argv[2]);
    if (minimum_out_degree_ratio < .5 || minimum_out_degree_ratio > 1) {
        printf("minimum out degree ratio must be between .5 and 1 inclusive\n");
        return 1;
    }
    minimum_in_degree_ratio = atof(argv[3]);
    if (minimum_in_degree_ratio < .5 || minimum_in_degree_ratio > 1) {
        printf("minimum in degree ratio must be between .5 and 1 inclusive\n");
        return 1;
    }
    minimum_clique_size = atoi(argv[4]);
    if (minimum_clique_size <= 1) {
        printf("minimum size must be greater than 1\n");
        return 1;
    }
    if (CPU_EXPAND_THRESHOLD > dss.EXPAND_THRESHOLD) {
        cout << "CPU_EXPAND_THRESHOLD must be less than the EXPAND_THRESHOLD" << endl;
        return 1;
    }

    // DEBUG
    output = argv[6];
    filename = "DQC-O10_" + output;
    output_file.open(filename);
    if (dss.DEBUG_TOGGLE) {
        output_file << endl << ">:OUTPUT FROM P1 PROCESS 0: " << endl << endl;
        //initialize_maxes();
    }

    // TIME
    auto start = chrono::high_resolution_clock::now();

    // GRAPH / MINDEGS
    cout << ">:PROGRAM 1 START" << endl << ">:PRE-PROCESSING" << endl;
    CPU_Graph hg(read_file);
    read_file.close();
    // CURSOR - program works up to this point

    // calculate_minimum_degrees(hg, minimum_out_degrees, minimum_in_degrees, minimum_out_degree_ratio, minimum_in_degree_ratio);
    // filename = "DQC-TZ0_" + output;
    // write_file.open(filename);

    // // TIME
    // auto stop = chrono::high_resolution_clock::now();
    // auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    // cout << ">:LOADING TIME: " << duration.count() << " ms" << endl;

    // // SEARCH
    // p1_search(hg, write_file, dss, minimum_degrees, minimum_degree_ratio, minimum_clique_size, output);

    // write_file.close();

    // // DEBUG
    // if (dss.DEBUG_TOGGLE) {
    //     print_maxes();
    //     output_file << endl;
    // }
    // output_file.close();

    // auto stop2 = chrono::high_resolution_clock::now();
    // auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2);
    // cout << ">:TOTAL TIME: " << duration2.count() << " ms" << endl;
    // cout << ">:PROGRAM 1 END" << endl;

    return 0;
}