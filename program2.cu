#include "./inc/common.hpp"
#include "./inc/host_functions.hpp"
#include "./inc/host_debug.h"

// MAIN
int main(int argc, char* argv[])
{
    // TIME
    auto start2 = chrono::high_resolution_clock::now();

    double minimum_degree_ratio;        // connection requirement for cliques
    int minimum_clique_size;            // minimum size for cliques
    int* minimum_degrees;               // stores the minimum connections per vertex for all size cliques
    int world_size;                     // number of cpu processes
    int world_rank;                     // current cpu processes rank
    string filename;                    // used in concatenation for making filenames
    string filename2;                   // used for second filename in remove non max
    ifstream read_file;                 // multiple read files
    ofstream write_file;                // writing results to mutiple files
    string line;                        // stores lines from read file
    string output;                      // distinct output name so programs can be run simultaneously

    // ENSURE PROPER USAGE
    if (argc != 6) {
        printf("Usage: ./program2 <graph_file> <gamma> <min_size> <ds_sizes_file> <output_file>\n");
        return 1;
    }
    read_file.open(argv[4], ios::in);
    if(!read_file.is_open()){
        cout << "invalid data structure sizes file\n" << endl;
    }
    read_file.close();
    // reads the sizes of the data structures
    DS_Sizes dss(argv[4]);
    read_file.open(argv[1], ios::in);
    if (!read_file.is_open()) {
        printf("invalid graph file\n");
        return 1;
    }
    minimum_degree_ratio = atof(argv[2]);
    if (minimum_degree_ratio < .5 || minimum_degree_ratio > 1) {
        printf("minimum degree ratio must be between .5 and 1 inclusive\n");
        return 1;
    }
    minimum_clique_size = atoi(argv[3]);
    if (minimum_clique_size <= 1) {
        printf("minimum size must be greater than 1\n");
        return 1;
    }
    if (CPU_EXPAND_THRESHOLD > dss.expand_threshold) {
        cout << "CPU_EXPAND_THRESHOLD must be less than the EXPAND_THRESHOLD" << endl;
        return 1;
    }

    // MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    wsize = world_size;
    grank = world_rank;

    // DEBUG
    output = argv[5];
    filename = "o_" + output + "_p2_" + to_string(grank) + ".txt";
    output_file.open(filename);
    if (dss.debug_toggle) {
        output_file << endl << ">:OUTPUT FROM P2 PROCESS: " << grank << endl << endl;
        initialize_maxes();
    }

    // TIME
    auto start = chrono::high_resolution_clock::now();

    // GRAPH / MINDEGS
    if(grank == 0){
        cout << ">:PROGRAM 2 START" << endl << ">:PRE-PROCESSING" << endl;
    }
    CPU_Graph hg(read_file);
    read_file.close();
    calculate_minimum_degrees(hg, minimum_degrees, minimum_degree_ratio);
    filename = "t_" + output + "_" + to_string(grank) + ".txt";
    write_file.open(filename, ios::app);

    // TIME
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    if(grank == 0){
        cout << ">:LOADING TIME: " << duration.count() << " ms" << endl;
    }

    // SEARCH
    p2_search(hg, write_file, dss, minimum_degrees, minimum_degree_ratio, minimum_clique_size, output);

    write_file.close();

    // DEBUG
    if (dss.debug_toggle) {
        print_maxes();
        if(grank == NUMBER_OF_PROCESSESS - 1){
            output_file << endl;
        }
    }
    output_file.close();

    // TIME
    auto stop2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2);
    if(grank == 0){
        cout << ">:TOTAL TIME: " << duration2.count() << " ms" << endl;
        cout << ">:PROGRAM 2 END" << endl;
    }

    MPI_Finalize();
    return 0;
}