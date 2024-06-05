#include "../inc/common.h"
#include "../inc/cuTS_MPI.h"

/*
TYPES OF MESSAGES AND THEIR MEANING:
c - taker thread asking for confirmation from giver thread
r - giver thread is requesting another thread to help take some of its work
C - giver thread giving confirmation to taker that it will send work
t - taker thread letting others know it has found worksss
f - taker thread letting others know it is free and could recieve work
D - giver thread declining to confirm to a taker that it will send work
* z - not an actual message but used as a place holder to indicate that
*/

// counts how many threads are done working in the program, if all threads are done can end the program
int count_free_list() {
    int cnt = 0;
    for (int i = 0; i < wsize; ++i) {
        if (global_free_list[i]) {
            cnt++;
        }
    }
    return cnt;
}

// asynchronously recieve a 1 char message from src thread in msg_buffer with a handle rq_recv_msg
void mpi_irecv(int src) {
    MPI_Irecv(msg_buffer[src], 1, MPI_CHAR, src, 0, MPI_COMM_WORLD,
              &rq_recv_msg[src]);

}

// asynchronously send a 1 char message to dest thread in msg_buffer with a handle rq_send_msg
void mpi_isend(int dest, char *msg) {
    //MPI_Isend(msg, strlen(msg) + 1, MPI_CHAR, dest, 0, MPI_COMM_WORLD,
    MPI_Isend(msg, 1, MPI_CHAR, dest, 0, MPI_COMM_WORLD,
              &rq_send_msg[dest]);

}

// asynchronously recieves messages from all other threads in world
void mpi_irecv_all(int rank) {
    for (int i = 0; i < wsize; i++) {
        if (i != rank) {
            mpi_irecv(i);
        }
    }
}

// asynchronously sends messages to all other threads in world
void mpi_isend_all(int rank, char *msg) {
    for (int i = 0; i < wsize; i++) {
        if (i != rank) {
            mpi_isend(i, msg);
        }
    }
}

// attempts to recieve work from another thread, returns true if work recieved else false
bool take_work(int from, int rank, unsigned int *buffer) {
    /// first ask the other node to confirm that it has pending work
    /// it might have finished it by the time we received the processing request or
    /// someone else might have offered it help

    // asks for confirmation form thread "from"
    mpi_isend(from, "c");

    MPI_Status status;

    // initialize last message to default value
    char last_msg = 'r';

    // keep getting messages from "from" thread until a recieved message is no longer a request message
    while (last_msg == 'r') // the while loop ensures that multiple `r' requests are removed
    {
        MPI_Wait(&rq_recv_msg[from], &status); //blocking wait till we get a proper response
        last_msg = msg_buffer[from][0];
        mpi_irecv(from); ///initiate a request
    }

    // when repsponce is recieved check whether from has work or not
    // if there is work on from
    if (last_msg == 'C') {

        // let other threads know this thread found work
        mpi_isend_all(rank, "t");

        // initialize a new MPI dataype of 5 unsigned ints called dt_point
        MPI_Datatype dt_point;
        MPI_Type_contiguous(5, MPI_UNSIGNED, &dt_point);
        MPI_Type_commit(&dt_point);

        // recieve MAXGB dt_points from the "from" thread, these dt_points will be interpreted as Point*
        // NOTE - this seems to be their data format, our equivalent would probably be Vertex
        MPI_Recv((Point *) buffer, MAXGB, dt_point, from, 1, MPI_COMM_WORLD, &status);
        
        // set current rank in free list as false
        global_free_list[rank] = false;

        return true;

        // if there is no more work on from
    } else if (last_msg == 'f') {

        // set from thread in free list as true
        global_free_list[from] = true;
    }

    return false;
}

// sends message to all other threads indicating it can recieve work, when it find a giver thread it requests confirmation
int take_work_wrap(int rank, unsigned int *buffer) {

    bool took_work = false;

    // send message to all threads that current thread is free
    mpi_isend_all(rank, "f");

    // set index in free list as true
    global_free_list[rank] = true;

    // until current thread has taken work from another or all threads are done
    while (!took_work && count_free_list() < wsize)
    {

        // for all other threads
        for (int i = 0; i < wsize; i++) {
            if (i == rank) {
                continue;
            }

            // does nothing, just needed for test call
            MPI_Status status;

            // initialize flag and last message to default values
            int flag = 1;
            char last_msg = 'z'; //invalid message

            // while we have still recieved a message from another thread i
            while (flag == 1)
            {
                // get the next message from thread i
                MPI_Test(&rq_recv_msg[i], &flag, &status); //check if we recvd a msg

                // if there was a message
                if (flag) {

                    // get the message
                    last_msg = msg_buffer[i][0];

                    // try to get he next message
                    mpi_irecv(i);

                    // if the message was that thread i was empty mark it as such
                    if (last_msg == 'f') {
                        global_free_list[i] = true;

                    // uf the message was that thread i got new work mark it as such
                    } else if (last_msg == 't') {
                        global_free_list[i] = false;
                    }
                }
            }
            if (last_msg == 'r')//someone is asking us to help to process their request...
            {

                // have to check whether we've taken work because, when we do we will still check messages form other threads
                // this is to see if they are also reporting free
                if (!took_work) {
                    took_work = take_work(i, rank, buffer);
                }
            }
        }
    }

    // return how many threads are done
    return count_free_list();
}

// actually sends work from current thread to "taker" thread
void give_work(int rank, int taker, unsigned int *buffer) {

    // not used, just needed as parameter
    MPI_Status status;

    // send C as indicating confirmation to taker thread
    mpi_isend(taker, "C");

    // wait until C is fully sent
    MPI_Wait(&rq_send_msg[taker], &status);
    /// At this point we know that the taker is waiting to recv data
    /// TODO WRITE CODE HERE to initiate data transfer
    /// USE TAG 1 for sync

    unsigned int giveSize;

    // UNSURE - how much is this, will be important for our program as well
    giveSize = ((buffer[0] + buffer[1])*2 + 2) / 5 + 1;

    // declare new MPI data type
    MPI_Datatype dt_point;
    MPI_Type_contiguous(5, MPI_UNSIGNED, &dt_point);
    MPI_Type_commit(&dt_point);

    // send data
    MPI_Send((Point *) buffer, giveSize, dt_point, taker, 1, MPI_COMM_WORLD);
}

// looks to see if another thread is requesting confirmation for transfer, if so transfers data to it and declines other threads askign for data
// the taker parameter is really a return value of the thread id of the thread we gave work to
bool check_for_confirmation(int rank, int &taker, unsigned int *buffer) {

    bool agreed_to_split_work = false;

    /// first try to respond all nodes which has send a confirmation request as all of them will be waiting
    // iterate through all other threads
    for (int i = 0; i < wsize; i++) {
        if (i == rank) {
            continue;
        }

        // not used, needed as parameter
        MPI_Status status;

        // initialize loop parameters
        int flag = true;
        char last_msg = 'z'; //invalid message

        // while there are still messages from thread i
        while (flag) /// move forward till we find the last message
        {

            // check if the current thread has recieved a message from thread i
            MPI_Test(&rq_recv_msg[i], &flag, &status); //check if we recvd a msg

            // if we recieved a message
            if (flag) {

                // get the last message from thread i
                last_msg = msg_buffer[i][0];

                // if the last message is f then mark thread i as free in the free list
                if (last_msg == 'f') {
                    global_free_list[i] = true;

                // if the last message is that the thread has taken work mark thread i as not free in the free list
                } else if (last_msg == 't') {
                    global_free_list[i] = false;
                }

                // recieve the next message
                mpi_irecv(i); /// initiate new recv request again
            }
        }

        // if the last message from some taker thread was asking for for data
        if (last_msg == 'c') //we found someone waiting for confirmation
        {

            // and we haven't given work to another thread yet
            if (!agreed_to_split_work) {

                // give work to the taker thread
                give_work(rank, i, buffer); //give work to this node
                agreed_to_split_work = true;

                // set the return variable taker as the id of thread we gave the work to
                taker = i;

            // we have already given work to another thread
            } else {

                ///send decline
                // send a declination message to the taker thread asking for data
                mpi_isend(i, "D");
            }
        }
    }

    // return whether we were able to give work to someone else
    // NOTE - I don't think we will need this with how our adaptation might work
    return agreed_to_split_work;
}

// check to see if a previous request for help was responded to, then send  another request for help and see if anyone repsonds
bool give_work_wrapper(int rank, int &taker, unsigned int *buffer) {

    // see if any thread has asked for confirmation from a previously sent request, if so this method also sends the data
    bool agreed_to_split_work = check_for_confirmation(rank, taker, buffer);


    /// no one send confirmation
    // if no one has sent a confirmation previously
    if (!agreed_to_split_work) {

        // for all other threads if they are currently free send a request for help
        for (int i = 0; i < wsize; i++) /// send a process request to all free nodes
        {
            if (i != rank && global_free_list[i]) {

                // the message for requesting for help
                mpi_isend(i, "r");
            }
        }

        /// retry to see someone send confirmation
        // now that new messages have been sent see if any thread is asking for confirmation
        agreed_to_split_work = check_for_confirmation(rank, taker, buffer);
    }

    // return whether work was split with another thread
    return agreed_to_split_work;
}

// TODO - this should encode part of buffer
// UNSURE - not sure how their data encoding works, don't think it is important as we will send our data differently either way
// seems like the first three elements are size data and the rest is the actual data
void encode_com_buffer(unsigned int *mpi_buffer,S_pointers s,unsigned iter,unsigned int buf_len){
    unsigned int pre_len = s.lengths[iter - 1];
    mpi_buffer[0] = pre_len;
    mpi_buffer[1] = buf_len;
    mpi_buffer[2] = iter;
    unsigned int copy_offset = 3;
    chkerr(cudaMemcpy(&mpi_buffer[copy_offset], s.results_table,pre_len * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    copy_offset+=(pre_len);
    chkerr(cudaMemcpy(&mpi_buffer[copy_offset], &s.results_table[pre_len+buf_len],
                      buf_len * sizeof(unsigned int),cudaMemcpyDeviceToHost));
    copy_offset+=buf_len;
    chkerr(cudaMemcpy(&mpi_buffer[copy_offset],s.indexes_table,pre_len * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    copy_offset+=pre_len;
    chkerr(cudaMemcpy(&mpi_buffer[copy_offset],&s.indexes_table[pre_len+buf_len],
                      buf_len * sizeof(unsigned int),cudaMemcpyDeviceToHost));
}

// TODO - this should decode part of buffer
// UNSURE - not sure how their data encoding works, don't think it is important as we will send our data differently either way
// seems like the first three elements are size data and the rest is the actual data
unsigned int decode_com_buffer(unsigned int *mpi_buffer,S_pointers &s){
    unsigned int pre_len = mpi_buffer[0];
    unsigned int buf_len = mpi_buffer[1];
    unsigned int iter = mpi_buffer[2];
    unsigned int copy_offset = 3;
    chkerr(cudaMemcpy(s.results_table,&mpi_buffer[copy_offset],(pre_len+buf_len) * sizeof(unsigned int),
                      cudaMemcpyHostToDevice));
    copy_offset+=(pre_len+buf_len);
    chkerr(cudaMemcpy(s.indexes_table,&mpi_buffer[copy_offset],(pre_len+buf_len) * sizeof(unsigned int),
                      cudaMemcpyHostToDevice));
    s.lengths[iter - 1] = pre_len;
    s.lengths[iter] = s.lengths[iter - 1] + buf_len;
    return iter;
}