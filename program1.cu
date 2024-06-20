#include "./inc/common.h"
#include "./inc/host_functions.h"
#include "./inc/host_debug.h"
#include "./inc/Quick_rmnonmax.h"

// TODO
// - see whether it's possible to parallelize some of calculate_LU_bounds
// - what is the bets mpi parameters, when should we split, how much, how often
// - do we need dumping cliques on the gpu?
// - add mem for slurm for p1
// - add serialization of debug maxes
// - dont need rmnonmax for p1
// - move programs to src folder

// CURSOR - do TODOs

// MAIN
int main(int argc, char* argv[])
{
    // TIME
    auto start2 = chrono::high_resolution_clock::now();

    double minimum_degree_ratio;        // connection requirement for cliques
    int minimum_clique_size;            // minimum size for cliques
    int* minimum_degrees;               // stores the minimum connections per vertex for all size cliques
    DS_Sizes dss("DS_Sizes.csv");       // reads the sizes of the data structures
    string filename;                    // used in concatenation for making filenames
    ifstream read_file;                 // multiple read files
    ofstream write_file;                // writing results to mutiple files
    string line;                        // stores lines from read file

    // ENSURE PROPER USAGE
    if (argc != 4) {
        printf("Usage: ./main <graph_file> <gamma> <min_size>\n");
        return 1;
    }
    // DEBUG - rm
    ifstream graph_stream(argv[1], ios::in);
    if (!read_file.is_open()) {
        printf("invalid graph file\n");
        return 1;
    }
    minimum_degree_ratio = atof(argv[2]);
    if (minimum_degree_ratio < .5 || minimum_degree_ratio>1) {
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

    // DEBUG
    filename = "output_DcuQC_p1_0.txt";
    output_file.open(filename);
    if (DEBUG_TOGGLE) {
        output_file << endl << ">:OUTPUT FROM P1 PROCESS 0: " << endl << endl;
        initialize_maxes();
    }

    // TIME
    auto start = chrono::high_resolution_clock::now();

    // GRAPH / MINDEGS
    cout << ">:PROGRAM 1 START" << endl << ">:PRE-PROCESSING" << endl;
    CPU_Graph hg(graph_stream);
    // DEBUG - rm
    cout << "graph" << endl;
    graph_stream.close();
    calculate_minimum_degrees(hg, minimum_degrees, minimum_degree_ratio);
    // DEBUG - rm
    cout << "mindegs" << endl;
    filename = "temp_DcuQC_0.txt";
    write_file.open(filename);

    // TIME
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "--->:LOADING TIME: " << duration.count() << " ms" << endl;

    // SEARCH
    p1_search(hg, write_file, dss, minimum_degrees, minimum_degree_ratio, minimum_clique_size);

    write_file.close();
    output_file.close();

    auto stop2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2);
    cout << "--->:TOTAL TIME: " << duration2.count() << " ms" << endl;
    cout << ">:PROGRAM 1 END" << endl;

    return 0;
}