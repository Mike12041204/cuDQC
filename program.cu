#include "./inc/common.hpp"
#include "./inc/host_functions.hpp"
#include "./inc/Quick_rmnonmax.h"
#include "./inc/host_debug.h"

// MAIN
int main(int argc, char* argv[])
{
    // --- HEADING ---

    double minimum_out_degree_ratio;
    double minimum_in_degree_ratio;
    int minimum_clique_size;            // minimum size for cliques
    int* minimum_out_degrees;           // stores the minimum connections per vertex for all size cliques
    int* minimum_in_degrees;
    string filename;                    // used in concatenation for making filenames
    string filename2;
    ifstream read_file;                 // multiple read files
    ofstream write_file;                // writing results to mutiple files
    string line;                        // stores lines from read file
    string output;
    int rank;
    int size;
    int num_cliques;

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

    // --- PRE-PROCESSING ---

    // TIME
    auto start = chrono::high_resolution_clock::now();

    // MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    wsize = size;
    grank = rank;

    // DEBUG
    output = argv[6];
    filename = "DQC-O" + to_string(grank) + "_" + output;
    output_file.open(filename);
    if (dss.DEBUG_TOGGLE) {
        output_file << "PROCESS: " << grank << endl << endl;
        initialize_maxes();
    }

    // TIME
    auto start1 = chrono::high_resolution_clock::now();

    // DEBUG
    if(grank == 0){
        cout << "LOADING GRAPH:       ";
    }

    // GRAPH
    CPU_Graph hg(read_file);
    // DEBUG
    if(grank == 0){
        output_file << "LOADED GRAPH" << endl;
        print_graph(hg);
    }
    read_file.close();

    // TIME
    auto stop1 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(stop1 - start1);
    if(grank == 0){
        cout << duration1.count() << " ms" << endl;
        cout << "CALCULATING MINDEGS: ";
    }
    start1 = chrono::high_resolution_clock::now();
    auto start2 = chrono::high_resolution_clock::now();

    // MINDEGS
    minimum_out_degrees = new int[hg.number_of_vertices + 1];
    minimum_in_degrees = new int[hg.number_of_vertices + 1];
    h_calculate_minimum_degrees(hg, minimum_out_degrees, minimum_out_degree_ratio);
    h_calculate_minimum_degrees(hg, minimum_in_degrees, minimum_in_degree_ratio);

    // TIME
    stop1 = chrono::high_resolution_clock::now();
    duration1 = chrono::duration_cast<chrono::milliseconds>(stop1 - start1);
    if(grank == 0){
        cout << duration1.count() << " ms" << endl;
    }

    // --- PROCESSING ---

    // OPEN RESULTS FILE
    filename = "DQC-T" + to_string(grank) + "_" + output;
    write_file.open(filename);

    // SEARCH
    h_search(hg, write_file, dss, minimum_out_degrees, minimum_in_degrees, minimum_out_degree_ratio, minimum_in_degree_ratio, minimum_clique_size, output);

    write_file.close();

    // ensure all temp files are written and closed
    MPI_Barrier(MPI_COMM_WORLD);

    // TIME
    auto stop2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2);
    auto computation_time = duration2.count();

    // --- POST-PROCESSING ---

    if(grank == 0){
        // TIME
        start1 = chrono::high_resolution_clock::now();
        cout << "REMOVING NON-MAX:    ";

        // COMBINE RESULTS 
        filename = "DQC-T_" + output;
        write_file.open(filename);
        for (int i = 0; i < NUMBER_OF_PROCESSESS; ++i) {
            filename = "DQC-T" + to_string(i) + "_" + output;
            read_file.open(filename);
            while (getline(read_file, line)) {
                write_file << line << endl;
            }
            read_file.close();
        }        

        // RM NON-MAX
        if(!(write_file.tellp() == ofstream::pos_type(0))){
            filename = "DQC-T_" + output;
            filename2 = "DQC-R_" + output;
            num_cliques = RemoveNonMax(filename.c_str(), filename2.c_str());
        }
        else{
            num_cliques = 0;
        }
        write_file.close();

        // TIME
        stop1 = chrono::high_resolution_clock::now();
        duration1 = chrono::duration_cast<chrono::milliseconds>(stop1 - start1);
        cout << duration1.count() << " ms" << endl;
    }

    // DEBUG
    if (dss.DEBUG_TOGGLE) {
        print_maxes();
        output_file << endl;
    }
    output_file.close();

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    if(grank == 0){
        cout << "TOTAL TIME:          " << duration.count() << " ms" << endl << endl;
        cout << "COMPUTATION TIME:    " << computation_time << "ms" << endl;
        cout << "NUMBER OF CLIQUES:   " << num_cliques << endl;
    }

    delete[] minimum_out_degrees;
    delete[] minimum_in_degrees;
    MPI_Finalize();
    return 0;
}