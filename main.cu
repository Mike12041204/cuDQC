#include "./inc/common.h"
#include "./inc/host_functions.h"
#include "./inc/host_debug.h"
#include "./inc/Quick_rmnonmax.h"

// TODO
// - reevaluate and change where uint64_t's are used
// - label for vertices can be a byte rather than int
// - don't need lvl2adj in all places anymore
// - look for places where we can break early
// - examine code for unnecessary syncs on the GPU
// - in degree pruning see if we can remove failed_found by consolidating with success
// - see whether it's possible to parallelize some of calculate_LU_bounds
// - dont need 2 tasks on GPU
// - make global variables local
// - make cuTS mpi its own file
// - review all code and code style

// MAIN
int main(int argc, char* argv[])
{
    double minimum_degree_ratio;
    int minimum_clique_size;
    int* minimum_degrees;
    int scheduling_toggle;
    DS_Sizes dss("DS_Sizes.csv");

    // TIME
    auto start2 = chrono::high_resolution_clock::now();

    // ENSURE PROPER USAGE
    if (argc != 4) {
        printf("Usage: ./main <graph_file> <gamma> <min_size>\n");
        return 1;
    }
    ifstream graph_stream(argv[1], ios::in);
    if (!graph_stream.is_open()) {
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

    // MPI
    MPI_Init(&argc,&argv);
    // number of cpu threads
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    wsize = world_size;
    // current cpu threads rank
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    grank = world_rank;

    // DEBUG
    string output_filename = "output_DcuQC_" + to_string(grank) + ".txt";
    ofstream output_file(output_filename);
    if (DEBUG_TOGGLE) {
        output_file << "Output from process " << grank << endl << endl;
        initialize_maxes();
    }

    // TIME
    auto start = chrono::high_resolution_clock::now();

    // GRAPH / MINDEGS
    if(grank == 0){
        cout << ">:PRE-PROCESSING" << endl;
    }
    CPU_Graph hg(graph_stream);
    graph_stream.close();
    calculate_minimum_degrees(hg, minimum_degrees, minimum_degree_ratio);
    string temp_filename = "temp_DcuQC_" + to_string(grank) + ".txt";
    ofstream temp_results(temp_filename);

    // TIME
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    if(grank == 0){
        cout << "--->:LOADING TIME: " << duration.count() << " ms" << endl;
    }

    // SEARCH
    search(hg, temp_results, output_file, dss, minimum_degrees, minimum_degree_ratio, minimum_clique_size);

    temp_results.close();

    // DEBUG
    if (DEBUG_TOGGLE) {
        print_maxes(output_file);
    }
    output_file.close();

    // TIME
    auto start1 = chrono::high_resolution_clock::now();

    MPI_Barrier(MPI_COMM_WORLD);
    if(grank == 0){

        // COMBINE RESULTS
        ofstream all_temp("temp_DcuQC.txt");
        for (int i = 0; i < NUMBER_OF_PROCESSESS; ++i) {
            string temp_filename = "temp_DcuQC_" + to_string(i) + ".txt";
            ifstream temp_file(temp_filename);
            string line;
            while (getline(temp_file, line)) {
                all_temp << line << endl;
            }
            temp_file.close();
        }

        // Check if the temp file is empty
        bool temp_empty = false;
        if (all_temp.tellp() == ofstream::pos_type(0)) {
            temp_empty = true;
        }
        all_temp.close();

        // RM NON-MAX
        if(!temp_empty){
            RemoveNonMax("temp_DcuQC.txt", "results_DcuQC.txt");
        }
        else{
            cout << ">:NUMBER OF MAXIMAL CLIQUES: 0" << endl;
        }
    }

    // TIME
    auto stop1 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(stop1 - start1);
    if(grank == 0){
        cout << "--->:REMOVE NON-MAX TIME: " << duration1.count() << " ms" << endl;
    }
    auto stop2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2);
    if(grank == 0){
        cout << "--->:TOTAL TIME: " << duration2.count() << " ms" << endl;
        cout << ">:PROGRAM END" << endl;
    }

    MPI_Finalize();
    return 0;
}

// MOVE THIS CODE TO THE TOP OF THE PROGRAM WHEN IMPLEMENTING IT

// for every task there is a seperate message buffer and incoming/outgoing handle slot
char msg_buffer[NUMBER_OF_PROCESSESS][100];
// array of handles for messages with all other thread, allows for asynchronous messaging, handles say whether message is complete
MPI_Request rq_send_msg[NUMBER_OF_PROCESSESS];
MPI_Request rq_recv_msg[NUMBER_OF_PROCESSESS];
bool global_free_list[NUMBER_OF_PROCESSESS];

/*
TYPES OF MESSAGES AND THEIR MEANING:
c - taker thread asking for confirmation from giver thread
r - giver thread is requesting another thread to help take some of its work
C - giver thread giving confirmation to taker that it will send work
t - taker thread letting others know it has found worksss
f - taker thread letting others know it is free and could recieve work
D - giver thrread declining to confirm to a taker that it will send work
* z - not an actual message but used as a place holder to indicate that
*/

// // counts how many threads are done working in the program, if all threads are done can end the program
// int count_free_list() {
//     int cnt = 0;
//     for (int i = 0; i < wsize; ++i) {
//         if (global_free_list[i]) {
//             cnt++;
//         }
//     }
//     return cnt;
// }

// // asynchronously recieve a 1 char message from src thread in msg_buffer with a handle rq_recv_msg
// void mpi_irecv(int src) {
//     MPI_Irecv(msg_buffer[src], 1, MPI_CHAR, src, 0, MPI_COMM_WORLD,
//               &rq_recv_msg[src]);

// }

// // asynchronously send a 1 char message to dest thread in msg_buffer with a handle rq_send_msg
// void mpi_isend(int dest, char *msg) {
//     //MPI_Isend(msg, strlen(msg) + 1, MPI_CHAR, dest, 0, MPI_COMM_WORLD,
//     MPI_Isend(msg, 1, MPI_CHAR, dest, 0, MPI_COMM_WORLD,
//               &rq_send_msg[dest]);

// }

// // asynchronously recieves messages from all other threads in world
// void mpi_irecv_all(int rank) {
//     for (int i = 0; i < wsize; i++) {
//         if (i != rank) {
//             mpi_irecv(i);
//         }
//     }
// }

// // asynchronously sends messages to all other threads in world
// void mpi_isend_all(int rank, char *msg) {
//     for (int i = 0; i < wsize; i++) {
//         if (i != rank) {
//             mpi_isend(i, msg);
//         }
//     }
// }

// // attempts to recieve work from another thread, returns true if work recieved else false
// bool take_work(int from, int rank, unsigned int *buffer) {
//     /// first ask the other node to confirm that it has pending work
//     /// it might have finished it by the time we received the processing request or
//     /// someone else might have offered it help

//     // asks for confirmation form thread "from"
//     mpi_isend(from, "c");

//     MPI_Status status;

//     // initialize last message to default value
//     char last_msg = 'r';

//     // keep getting messages from "from" thread until a recieved message is no longer a request message
//     while (last_msg == 'r') // the while loop ensures that multiple `r' requests are removed
//     {
//         MPI_Wait(&rq_recv_msg[from], &status); //blocking wait till we get a proper response
//         last_msg = msg_buffer[from][0];
//         mpi_irecv(from); ///initiate a request
//     }

//     // when repsponce is recieved check whether from has work or not
//     // if there is work on from
//     if (last_msg == 'C') {

//         // let other threads know this thread found work
//         mpi_isend_all(rank, "t");

//         // initialize a new MPI dataype of 5 unsigned ints called dt_point
//         MPI_Datatype dt_point;
//         MPI_Type_contiguous(5, MPI_UNSIGNED, &dt_point);
//         MPI_Type_commit(&dt_point);

//         // recieve MAXGB dt_points from the "from" thread, these dt_points will be interpreted as Point*
//         // NOTE - this seems to be their data format, our equivalent would probably be Vertex
//         MPI_Recv((Point *) buffer, MAXGB, dt_point, from, 1, MPI_COMM_WORLD, &status);
        
//         // set current rank in free list as false
//         global_free_list[rank] = false;

//         return true;

//         // if there is no more work on from
//     } else if (last_msg == 'f') {

//         // set from thread in free list as true
//         global_free_list[from] = true;
//     }

//     return false;
// }

// // sends message to all other threads indicating it can recieve work, when it find a giver thread it requests confirmation
// int take_work_wrap(int rank, unsigned int *buffer) {

//     bool took_work = false;

//     // send message to all threads that current thread is free
//     mpi_isend_all(rank, "f");

//     // set index in free list as true
//     global_free_list[rank] = true;

//     // until current thread has taken work from another or all threads are done
//     while (!took_work && count_free_list() < wsize)
//     {

//         // for all other threads
//         for (int i = 0; i < wsize; i++) {
//             if (i == rank) {
//                 continue;
//             }

//             // does nothing, just needed for test call
//             MPI_Status status;

//             // initialize flag and last message to default values
//             int flag = 1;
//             char last_msg = 'z'; //invalid message

//             // while we have still recieved a message from another thread i
//             while (flag == 1)
//             {
//                 // get the next message from thread i
//                 MPI_Test(&rq_recv_msg[i], &flag, &status); //check if we recvd a msg

//                 // if there was a message
//                 if (flag) {

//                     // get the message
//                     last_msg = msg_buffer[i][0];

//                     // try to get he next message
//                     mpi_irecv(i);

//                     // if the message was that thread i was empty mark it as such
//                     if (last_msg == 'f') {
//                         global_free_list[i] = true;

//                     // uf the message was that thread i got new work mark it as such
//                     } else if (last_msg == 't') {
//                         global_free_list[i] = false;
//                     }
//                 }
//             }
//             if (last_msg == 'r')//someone is asking us to help to process their request...
//             {

//                 // have to check whether we've taken work because, when we do we will still check messages form other threads
//                 // this is to see if they are also reporting free
//                 if (!took_work) {
//                     took_work = take_work(i, rank, buffer);
//                 }
//             }
//         }
//     }

//     // return how many threads are done
//     return count_free_list();
// }

// // actually sends work from current thread to "taker" thread
// void give_work(int rank, int taker, unsigned int *buffer) {

//     // not used, just needed as parameter
//     MPI_Status status;

//     // send C as indicating confirmation to taker thread
//     mpi_isend(taker, "C");

//     // wait until C is fully sent
//     MPI_Wait(&rq_send_msg[taker], &status);
//     /// At this point we know that the taker is waiting to recv data
//     /// TODO WRITE CODE HERE to initiate data transfer
//     /// USE TAG 1 for sync

//     unsigned int giveSize;

//     // UNSURE - how much is this, will be important for our program as well
//     giveSize = ((buffer[0] + buffer[1])*2 + 2) / 5 + 1;

//     // declare new MPI data type
//     MPI_Datatype dt_point;
//     MPI_Type_contiguous(5, MPI_UNSIGNED, &dt_point);
//     MPI_Type_commit(&dt_point);

//     // send data
//     MPI_Send((Point *) buffer, giveSize, dt_point, taker, 1, MPI_COMM_WORLD);
// }

// // looks to see if another thread is requesting confirmation for transfer, if so transfers data to it and declines other threads askign for data
// // the taker parameter is really a return value of the thread id of the thread we gave work to
// bool check_for_confirmation(int rank, int &taker, unsigned int *buffer) {

//     bool agreed_to_split_work = false;

//     /// first try to respond all nodes which has send a confirmation request as all of them will be waiting
//     // iterate through all other threads
//     for (int i = 0; i < wsize; i++) {
//         if (i == rank) {
//             continue;
//         }

//         // not used, needed as parameter
//         MPI_Status status;

//         // initialize loop parameters
//         int flag = true;
//         char last_msg = 'z'; //invalid message

//         // while there are still messages from thread i
//         while (flag) /// move forward till we find the last message
//         {

//             // check if the current thread has recieved a message from thread i
//             MPI_Test(&rq_recv_msg[i], &flag, &status); //check if we recvd a msg

//             // if we recieved a message
//             if (flag) {

//                 // get the last message from thread i
//                 last_msg = msg_buffer[i][0];

//                 // if the last message is f then mark thread i as free in the free list
//                 if (last_msg == 'f') {
//                     global_free_list[i] = true;

//                 // if the last message is that the thread has taken work mark thread i as not free in the free list
//                 } else if (last_msg == 't') {
//                     global_free_list[i] = false;
//                 }

//                 // recieve the next message
//                 mpi_irecv(i); /// initiate new recv request again
//             }
//         }

//         // if the last message from some taker thread was asking for for data
//         if (last_msg == 'c') //we found someone waiting for confirmation
//         {

//             // and we haven't given work to another thread yet
//             if (!agreed_to_split_work) {

//                 // give work to the taker thread
//                 give_work(rank, i, buffer); //give work to this node
//                 agreed_to_split_work = true;

//                 // set the return variable taker as the id of thread we gave the work to
//                 taker = i;

//             // we have already given work to another thread
//             } else {

//                 ///send decline
//                 // send a declination message to the taker thread asking for data
//                 mpi_isend(i, "D");
//             }
//         }
//     }

//     // return whether we were able to give work to someone else
//     // NOTE - I don't think we will need this with how our adaptation might work
//     return agreed_to_split_work;
// }

// // check to see if a previous request for help was responded to, then send  another request for help and see if anyone repsonds
// bool give_work_wrapper(int rank, int &taker, unsigned int *buffer) {

//     // see if any thread has asked for confirmation from a previously sent request, if so this method also sends the data
//     bool agreed_to_split_work = check_for_confirmation(rank, taker, buffer);


//     /// no one send confirmation
//     // if no one has sent a confirmation previously
//     if (!agreed_to_split_work) {

//         // for all other threads if they are currently free send a request for help
//         for (int i = 0; i < wsize; i++) /// send a process request to all free nodes
//         {
//             if (i != rank && global_free_list[i]) {

//                 // the message for requesting for help
//                 mpi_isend(i, "r");
//             }
//         }

//         /// retry to see someone send confirmation
//         // now that new messages have been sent see if any thread is asking for confirmation
//         agreed_to_split_work = check_for_confirmation(rank, taker, buffer);
//     }

//     // return whether work was split with another thread
//     return agreed_to_split_work;
// }

// // TODO - this should encode part of buffer
// // UNSURE - not sure how their data encoding works, don't think it is important as we will send our data differently either way
// // seems like the first three elements are size data and the rest is the actual data
// void encode_com_buffer(unsigned int *mpi_buffer,S_pointers s,unsigned iter,unsigned int buf_len){
//     unsigned int pre_len = s.lengths[iter - 1];
//     mpi_buffer[0] = pre_len;
//     mpi_buffer[1] = buf_len;
//     mpi_buffer[2] = iter;
//     unsigned int copy_offset = 3;
//     chkerr(cudaMemcpy(&mpi_buffer[copy_offset], s.results_table,pre_len * sizeof(unsigned int),
//                       cudaMemcpyDeviceToHost));
//     copy_offset+=(pre_len);
//     chkerr(cudaMemcpy(&mpi_buffer[copy_offset], &s.results_table[pre_len+buf_len],
//                       buf_len * sizeof(unsigned int),cudaMemcpyDeviceToHost));
//     copy_offset+=buf_len;
//     chkerr(cudaMemcpy(&mpi_buffer[copy_offset],s.indexes_table,pre_len * sizeof(unsigned int),
//                       cudaMemcpyDeviceToHost));
//     copy_offset+=pre_len;
//     chkerr(cudaMemcpy(&mpi_buffer[copy_offset],&s.indexes_table[pre_len+buf_len],
//                       buf_len * sizeof(unsigned int),cudaMemcpyDeviceToHost));
// }

// // TODO - this should decode part of buffer
// // UNSURE - not sure how their data encoding works, don't think it is important as we will send our data differently either way
// // seems like the first three elements are size data and the rest is the actual data
// unsigned int decode_com_buffer(unsigned int *mpi_buffer,S_pointers &s){
//     unsigned int pre_len = mpi_buffer[0];
//     unsigned int buf_len = mpi_buffer[1];
//     unsigned int iter = mpi_buffer[2];
//     unsigned int copy_offset = 3;
//     chkerr(cudaMemcpy(s.results_table,&mpi_buffer[copy_offset],(pre_len+buf_len) * sizeof(unsigned int),
//                       cudaMemcpyHostToDevice));
//     copy_offset+=(pre_len+buf_len);
//     chkerr(cudaMemcpy(s.indexes_table,&mpi_buffer[copy_offset],(pre_len+buf_len) * sizeof(unsigned int),
//                       cudaMemcpyHostToDevice));
//     s.lengths[iter - 1] = pre_len;
//     s.lengths[iter] = s.lengths[iter - 1] + buf_len;
//     return iter;
// }