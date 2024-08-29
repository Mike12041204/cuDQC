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
using namespace std;

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
    void write_serialized(char* output_file);
};

int h_sort_asce(const void* a, const void* b);
void print_CPU_Graph(CPU_Graph& hg);

// MAIN
int main(int argc, char* argv[])
{
    // ENSURE PROPER USAGE
    if (argc != 3) {
        printf("Usage: ./main <graph_file> <output_file>\n");
        return 1;
    }
    ifstream graph_stream(argv[1], ios::in);
    if (!graph_stream.is_open()) {
        printf("invalid graph file\n");
        return 1;
    }

    // GRAPH
    CPU_Graph hg(graph_stream);
    hg.write_serialized(argv[2]);
    graph_stream.close();

    // DEBUG- rm
    print_CPU_Graph(hg);
    
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

void print_CPU_Graph(CPU_Graph& hg) {
    cout << endl << " --- (CPU_Graph)host_graph details --- " << endl;
    cout << endl << "|V|: " << hg.number_of_vertices << " |E|: " << hg.number_of_edges << endl;
    cout << endl << "Out Offsets:" << endl;
    for (uint64_t i = 0; i <= hg.number_of_vertices; i++) {
        cout << hg.out_offsets[i] << " ";
    }
    cout << endl << "Out Neighbors:" << endl;
    for (uint64_t i = 0; i < hg.number_of_edges; i++) {
        cout << hg.out_neighbors[i] << " ";
    }
    cout << endl << "In Offsets:" << endl;
    for (uint64_t i = 0; i <= hg.number_of_vertices; i++) {
        cout << hg.in_offsets[i] << " ";
    }
    cout << endl << "In Neighbors:" << endl;
    for (uint64_t i = 0; i < hg.number_of_edges; i++) {
        cout << hg.in_neighbors[i] << " ";
    }
    // cout << endl << "Twohop Offsets:" << endl;
    // for (uint64_t i = 0; i <= hg.number_of_vertices; i++) {
    //     cout << hg.twohop_offsets[i] << " ";
    // }
    // cout << endl << "Twohop Neighbors:" << endl;
    // for (uint64_t i = 0; i < hg.number_of_lvl2adj; i++) {
    //     cout << hg.twohop_neighbors[i] << " ";
    // }
    // cout << endl << endl;
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

    // reset infile
    graph_stream.clear();
    graph_stream.seekg(0);

    // read all 1hop adj
    current_line = 0;
    number_of_edges = 0;
    while (getline(graph_stream, line)) {
        line_stream.clear();
        line_stream.str(line);
        cout << line << "!!!" << endl;

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

    // graph_stream.seekg(0, graph_stream.end);
    // string graph_text(graph_stream.tellg(), 0);
    // graph_stream.seekg(0);
    // graph_stream.read(const_cast<char*>(graph_text.data()), graph_text.size());

    // out_offsets = new uint64_t[OFFSETS_SIZE];
    // out_neighbors = new int[LVL1ADJ_SIZE];
    // in_offsets = new uint64_t[OFFSETS_SIZE];
    // in_neighbors = new int[LVL1ADJ_SIZE];
    // twohop_neighbors = new int[LVL2ADJ_SIZE];

    // out_offsets[0] = 0;
    // in_offsets[0] = 0;
    // number_of_lvl2adj = 0;

    // int vertex_count = 0;
    // int number_count = 0;
    // int current_number = 0;
    // bool empty = true;



    // // TODO - way to detect and handle these cases without changing code?
    // // TWO FORMATS SO FAR
    // // 1 -  VSCode \r\n between lines, no ending character
    // // 2 - Visual Studio \n between lines, numerous \0 ending characters

    // // parse graph file assume adj are seperated by spaces ' ' and vertices are seperated by newlines "\r\n"
    // for (int i = 0; i < graph_text.size(); i++) {
    //     char character = graph_text[i];

    //     // line depends on whether newline is "\r\n" or '\n'
    //     if (character == '\n') {
    //         if (!empty) {
    //             onehop_neighbors[number_count++] = current_number;
    //         }
    //         onehop_offsets[++vertex_count] = number_count;
    //         current_number = 0;
    //         // line depends on whether newline is "\r\n" or '\n'
    //         //i++;
    //         empty = true;
    //     }
    //     else if (character == ' ') {
    //         onehop_neighbors[number_count++] = current_number;
    //         current_number = 0;
    //     }
    //     else if (character == '\0') {
    //         // line depends on whether newline is "\r\n" or '\n'
    //         break;
    //     }
    //     else {
    //         current_number = current_number * 10 + (graph_text[i] - '0');
    //         empty = false;
    //     }
    // }

    // // line depends on whether newline is "\r\n" or '\n'
    // // handle last element
    // if (!empty) {
    //     onehop_neighbors[number_count++] = current_number;
    // }
    // onehop_offsets[++vertex_count] = number_count;

    // // set variables and initialize twohop arrays
    // number_of_vertices = vertex_count;
    // number_of_edges = number_count;



    // twohop_offsets = new uint64_t[number_of_vertices + 1];

    // twohop_offsets[0] = 0;

    // bool* twohop_flag_DIA;
    // twohop_flag_DIA = new bool[number_of_vertices];
    // memset(twohop_flag_DIA, true, number_of_vertices * sizeof(bool));

    // // handle lvl2 adj
    // for (int i = 0; i < vertex_count; i++) {
    //     for (int j = onehop_offsets[i]; j < onehop_offsets[i + 1]; j++) {
    //         int lvl1adj = onehop_neighbors[j];
    //         if (twohop_flag_DIA[lvl1adj]) {
    //             twohop_neighbors[number_of_lvl2adj++] = lvl1adj;
    //             twohop_flag_DIA[lvl1adj] = false;
    //         }

    //         for (int k = onehop_offsets[lvl1adj]; k < onehop_offsets[lvl1adj + 1]; k++) {
    //             int lvl2adj = onehop_neighbors[k];
    //             if (twohop_flag_DIA[lvl2adj] && lvl2adj != i) {
    //                 twohop_neighbors[number_of_lvl2adj++] = lvl2adj;
    //                 twohop_flag_DIA[lvl2adj] = false;
    //             }
    //         }
    //     }

    //     twohop_offsets[i + 1] = number_of_lvl2adj;

    //     for (int j = twohop_offsets[i]; j < twohop_offsets[i + 1]; j++) {
    //         twohop_flag_DIA[twohop_neighbors[j]] = true;
    //     }

    //     // sort adjacencies
    //     if (onehop_offsets[i + 1] != onehop_offsets[i]) {
    //         qsort(onehop_neighbors + onehop_offsets[i], onehop_offsets[i + 1] - onehop_offsets[i], sizeof(int), h_sort_asce);
    //     }
    //     if (twohop_offsets[i + 1] != twohop_offsets[i]) {
    //         qsort(twohop_neighbors + twohop_offsets[i], twohop_offsets[i + 1] - twohop_offsets[i], sizeof(int), h_sort_asce);
    //     }
    // }

    // delete twohop_flag_DIA;
}

void CPU_Graph::write_serialized(char* output_file)
{
    // ofstream out(output_file);

    // out << number_of_vertices << endl;
    // out << number_of_edges << endl;
    // out << number_of_lvl2adj << endl;

    // for (int i = 0; i < number_of_edges; i++) {
    //     out << onehop_neighbors[i];
    //     if (i < number_of_edges - 1) {
    //         out << " ";
    //     }
    // }
    // out << endl;

    // for (int i = 0; i < number_of_vertices + 1; i++) {
    //     out << onehop_offsets[i];
    //     if (i < number_of_vertices) {
    //         out << " ";
    //     }
    // }
    // out << endl;

    // for (int i = 0; i < number_of_lvl2adj; i++) {
    //     out << twohop_neighbors[i];
    //     if (i < number_of_lvl2adj - 1) {
    //         out << " ";
    //     }
    // }
    // out << endl;

    // for (int i = 0; i < number_of_vertices + 1; i++) {
    //     out << twohop_offsets[i];
    //     if (i < number_of_vertices) {
    //         out << " ";
    //     }
    // }
    // out << endl;

    // out.close();
}

CPU_Graph::~CPU_Graph()
{
    // delete onehop_neighbors;
    // delete onehop_offsets;
    // delete twohop_neighbors;
    // delete twohop_offsets;
}