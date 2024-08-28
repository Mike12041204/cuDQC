#include "./inc/common.h"
#include "./inc/Quick_rmnonmax.h"

// MAIN
int main(int argc, char* argv[])
{
    // TIME
    auto start1 = chrono::high_resolution_clock::now();

    string filename;                    // used in concatenation for making filenames
    string filename2;                   // used for second filename in remove non max
    ifstream read_file;                 // multiple read files
    ofstream write_file;                // writing results to mutiple files
    string line;                        // stores lines from read file
    string output;                      // distinct output name so programs can be run simultaneously

    // ENSURE PROPER USAGE
    if (argc != 2) {
        printf("Usage: ./program3 <output_file>\n");
        return 1;
    }
    output = argv[1];
    filename = "t_" + output + "_0.txt";
    read_file.open(filename, ios::in);
    if(!read_file.is_open()){
        cout << "invalid output file\n" << endl;
    }
    read_file.close();

    // TIME
    auto start3 = chrono::high_resolution_clock::now();

    // COMBINE RESULTS 
    cout << ">:PROGRAM 3 START" << endl << ">:COMBINING RESULTS" << endl;
    filename = "t_" + output + ".txt";
    write_file.open(filename);
    for (int i = 0; i < NUMBER_OF_PROCESSESS; ++i) {
        filename = "t_" + output + "_" + to_string(i) + ".txt";
        read_file.open(filename);
        while (getline(read_file, line)) {
            write_file << line << endl;
        }
        read_file.close();
    }

    // TIME
    auto stop3 = chrono::high_resolution_clock::now();
    auto duration3 = chrono::duration_cast<chrono::milliseconds>(stop3 - start3);
    cout << ">:COMBINE RESULTS TIME: " << duration3.count() << " ms" << endl;
    auto start2 = chrono::high_resolution_clock::now();

    // RM NON-MAX
    if(!(write_file.tellp() == ofstream::pos_type(0))){
        filename = "t_" + output + ".txt";
        filename2 = "r_" + output + ".txt";
        RemoveNonMax(filename.c_str(), filename2.c_str());
    }
    else{
        cout << "--->:NUMBER OF MAXIMAL CLIQUES: 0" << endl;
    }
    write_file.close();

    // TIME
    auto stop2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2);
    cout << ">:REMOVE NON-MAX TIME: " << duration2.count() << " ms" << endl;
    auto stop1 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(stop1 - start1);
    cout << ">:TOTAL TIME: " << duration1.count() << " ms" << endl;
    cout << ">:PROGRAM 3 END" << endl;
}