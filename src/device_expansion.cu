#include "../inc/common.h"
#include "../inc/device_expansion.h"
#include "../inc/device_helper.h"
#include "../inc/device_debug.h"

// returns 1 if lookahead succesful, 0 otherwise  
__device__ int d_lookahead_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    int pvertexid;
    int phelper1;
    int phelper2;

    if (LANE_IDX == 0) {
        wd.success[WIB_IDX] = true;
    }
    __syncwarp();

    // check if members meet degree requirement, dont need to check 2hop adj as diameter pruning guarentees all members will be within 2hops of eveything
    for (int i = LANE_IDX; i < wd.num_mem[WIB_IDX] && wd.success[WIB_IDX]; i += WARP_SIZE) {
        if (dd.read_vertices[wd.start[WIB_IDX] + i].indeg + dd.read_vertices[wd.start[WIB_IDX] + i].exdeg < dd.minimum_degrees[wd.tot_vert[WIB_IDX]]) {
            wd.success[WIB_IDX] = false;
            break;
        }
    }
    __syncwarp();

    if (!wd.success[WIB_IDX]) {
        return 0;
    }

    // update lvl2adj to candidates for all vertices
    for (int i = wd.num_mem[WIB_IDX] + LANE_IDX; i < wd.tot_vert[WIB_IDX]; i += WARP_SIZE) {
        pvertexid = dd.read_vertices[wd.start[WIB_IDX] + i].vertexid;
        
        for (int j = wd.num_mem[WIB_IDX]; j < wd.tot_vert[WIB_IDX]; j++) {
            if (j == i) {
                continue;
            }

            phelper1 = dd.read_vertices[wd.start[WIB_IDX] + j].vertexid;
            phelper2 = d_bsearch_array(dd.twohop_neighbors + dd.twohop_offsets[phelper1], dd.twohop_offsets[phelper1 + 1] - dd.twohop_offsets[phelper1], pvertexid);
        
            if (phelper2 > -1) {
                dd.read_vertices[wd.start[WIB_IDX] + i].lvl2adj++;
            }
        }
    }
    __syncwarp();

    // compares all vertices to the lemmas from Quick
    for (int j = wd.num_mem[WIB_IDX] + LANE_IDX; j < wd.tot_vert[WIB_IDX] && wd.success[WIB_IDX]; j += WARP_SIZE) {
        if (dd.read_vertices[wd.start[WIB_IDX] + j].lvl2adj < wd.num_cand[WIB_IDX] - 1 || dd.read_vertices[wd.start[WIB_IDX] + j].indeg + dd.read_vertices[wd.start[WIB_IDX] + j].exdeg < dd.minimum_degrees[wd.tot_vert[WIB_IDX]]) {
            wd.success[WIB_IDX] = false;
            break;
        }
    }
    __syncwarp();

    if (wd.success[WIB_IDX]) {
        // write to cliques
        uint64_t start_write = (WCLIQUES_SIZE * WARP_IDX) + dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * WARP_IDX) + (dd.wcliques_count[WARP_IDX])];
        for (int j = LANE_IDX; j < wd.tot_vert[WIB_IDX]; j += WARP_SIZE) {
            dd.wcliques_vertex[start_write + j] = dd.read_vertices[wd.start[WIB_IDX] + j].vertexid;
        }
        if (LANE_IDX == 0) {
            (dd.wcliques_count[WARP_IDX])++;
            dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * WARP_IDX) + (dd.wcliques_count[WARP_IDX])] = start_write - (WCLIQUES_SIZE * WARP_IDX) + wd.tot_vert[WIB_IDX];
        }
        return 1;
    }

    return 0;
}

// returns 1 if failed found after removing, 0 otherwise
__device__ int d_remove_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    int pvertexid;
    int phelper1;
    int phelper2;

    int mindeg;

    mindeg = d_get_mindeg(wd.num_mem[WIB_IDX], dd);

    // remove the last candidate in vertices
    if (LANE_IDX == 0) {
        wd.num_cand[WIB_IDX]--;
        wd.tot_vert[WIB_IDX]--;
        wd.success[WIB_IDX] = false;
    }
    __syncwarp();

    // update info of vertices connected to removed cand
    pvertexid = dd.read_vertices[wd.start[WIB_IDX] + wd.tot_vert[WIB_IDX]].vertexid;

    for (int i = LANE_IDX; i < wd.tot_vert[WIB_IDX] && !wd.success[WIB_IDX]; i += WARP_SIZE) {
        phelper1 = dd.read_vertices[wd.start[WIB_IDX] + i].vertexid;
        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[pvertexid], dd.onehop_offsets[pvertexid + 1] - dd.onehop_offsets[pvertexid], phelper1);

        if (phelper2 > -1) {
            dd.read_vertices[wd.start[WIB_IDX] + i].exdeg--;

            if (phelper1 < wd.num_mem[WIB_IDX] && dd.read_vertices[wd.start[WIB_IDX] + phelper1].indeg + dd.read_vertices[wd.start[WIB_IDX] + phelper1].exdeg < mindeg) {
                wd.success[WIB_IDX] = true;
                break;
            }
        }
    }
    __syncwarp();

    if (wd.success[WIB_IDX]) {
        return 1;
    }

    return 0;
}

// returns 1 if failed found or invalid bound, 0 otherwise
__device__ int d_add_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    int pvertexid;
    int phelper1;
    int phelper2;
    bool failed_found;



    // ADD ONE VERTEX
    pvertexid = ld.vertices[wd.number_of_members[WIB_IDX]].vertexid;

    if (LANE_IDX == 0) {
        ld.vertices[wd.number_of_members[WIB_IDX]].label = 1;
        wd.number_of_members[WIB_IDX]++;
        wd.number_of_candidates[WIB_IDX]--;
    }
    __syncwarp();

    for (int i = LANE_IDX; i < wd.tot_vert[WIB_IDX]; i += WARP_SIZE) {
        phelper1 = ld.vertices[i].vertexid;
        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[pvertexid], dd.onehop_offsets[pvertexid + 1] - dd.onehop_offsets[pvertexid], phelper1);

        if (phelper2 > -1) {
            ld.vertices[i].exdeg--;
            ld.vertices[i].indeg++;
        }
    }
    __syncwarp();



    // DIAMETER PRUNING
    d_diameter_pruning(dd, wd, ld, pvertexid);



    // DEGREE BASED PRUNING
    failed_found = d_degree_pruning(dd, wd, ld);

    // if vertex in x found as not extendable continue to next iteration
    if (failed_found) {
        return 1;
    }
   
    return 0;
}

// returns 2, if critical fail, 1 if failed found or invalid bound, 0 otherwise
__device__ int d_critical_vertex_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    // intersection
    int phelper1;

    // pruning
    int number_of_crit_adj;
    bool failed_found;



    // CRITICAL VERTEX PRUNING 
    // iterate through all vertices in clique
    for (int k = 0; k < wd.number_of_members[WIB_IDX]; k++) {

        // if they are a critical vertex
        if (ld.vertices[k].indeg + ld.vertices[k].exdeg == dd.minimum_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]] && ld.vertices[k].exdeg > 0) {
            phelper1 = ld.vertices[k].vertexid;

            // iterate through all candidates
            for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
                if (ld.vertices[i].label != 4) {
                    // if candidate is neighbor of critical vertex mark as such
                    if (d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], ld.vertices[i].vertexid) > -1) {
                        ld.vertices[i].label = 4;
                    }
                }
            }
        }
        __syncwarp();
    }



    // sort vertices so that critical vertex adjacent candidates are immediately after vertices within the clique
    d_sort(ld.vertices + wd.number_of_members[WIB_IDX], wd.number_of_candidates[WIB_IDX], d_sort_vert_cv);

    // count number of critical adjacent vertices
    number_of_crit_adj = 0;
    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
        if (ld.vertices[i].label == 4) {
            number_of_crit_adj++;
        }
        else {
            break;
        }
    }
    // get sum
    for (int i = 1; i < 32; i *= 2) {
        number_of_crit_adj += __shfl_xor_sync(0xFFFFFFFF, number_of_crit_adj, i);
    }



    failed_found = false;

    // reset adjacencies
    for (int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
        dd.adjacencies[(WVERTICES_SIZE * WARP_IDX) + i] = 0;
    }

    // if there were any neighbors of critical vertices
    if (number_of_crit_adj > 0)
    {
        // iterate through all vertices and update their degrees as if critical adjacencies were added and keep track of how many critical adjacencies they are adjacent to
        for (int k = LANE_IDX; k < wd.total_vertices[WIB_IDX]; k += WARP_SIZE) {
            phelper1 = ld.vertices[k].vertexid;

            for (int i = wd.number_of_members[WIB_IDX]; i < wd.number_of_members[WIB_IDX] + number_of_crit_adj; i++) {
                if (d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], ld.vertices[i].vertexid) > -1) {
                    ld.vertices[k].indeg++;
                    ld.vertices[k].exdeg--;
                }

                if (d_bsearch_array(dd.twohop_neighbors + dd.twohop_offsets[phelper1], dd.twohop_offsets[phelper1 + 1] - dd.twohop_offsets[phelper1], ld.vertices[i].vertexid) > -1) {
                    dd.adjacencies[(WVERTICES_SIZE * WARP_IDX) + k]++;
                }
            }
        }
        __syncwarp();

        // all vertices within the clique must be within 2hops of the newly added critical vertex adj vertices
        for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX]; k += WARP_SIZE) {
            if (dd.adjacencies[(WVERTICES_SIZE * WARP_IDX) + k] != number_of_crit_adj) {
                failed_found = true;
                break;
            }
        }
        failed_found = __any_sync(0xFFFFFFFF, failed_found);
        if (failed_found) {
            return 2;
        }

        // all critical adj vertices must all be within 2 hops of each other
        for (int k = wd.number_of_members[WIB_IDX] + LANE_IDX; k < wd.number_of_members[WIB_IDX] + number_of_crit_adj; k += WARP_SIZE) {
            if (dd.adjacencies[(WVERTICES_SIZE * WARP_IDX) + k] < number_of_crit_adj - 1) {
                failed_found = true;
                break;
            }
        }
        failed_found = __any_sync(0xFFFFFFFF, failed_found);
        if (failed_found) {
            return 2;
        }

        // no failed vertices found so add all critical vertex adj candidates to clique
        for (int k = wd.number_of_members[WIB_IDX] + LANE_IDX; k < wd.number_of_members[WIB_IDX] + number_of_crit_adj; k += WARP_SIZE) {
            ld.vertices[k].label = 1;
        }

        if (LANE_IDX == 0) {
            wd.number_of_members[WIB_IDX] += number_of_crit_adj;
            wd.number_of_candidates[WIB_IDX] -= number_of_crit_adj;
        }
        __syncwarp();
    }



    // DIAMTER PRUNING
    d_diameter_pruning_cv(dd, wd, ld, number_of_crit_adj);



    // DEGREE BASED PRUNING
    failed_found = d_degree_pruning(dd, wd, ld);

    // if vertex in x found as not extendable continue to next iteration
    if (failed_found) {
        return 1;
    }

    return 0;
}

// diameter pruning intitializes vertices labels and candidate indegs array for use in iterative degree pruning
__device__ void d_diameter_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int pvertexid)
{
    // vertices size * warp idx + (vertices size / warp size) * lane idx
    int lane_write = ((WVERTICES_SIZE * WARP_IDX) + ((WVERTICES_SIZE / WARP_SIZE) * LANE_IDX));

    // intersection
    int phelper1;
    int phelper2;

    // vertex iteration
    int lane_remaining_count;

    lane_remaining_count = 0;

    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
        ld.vertices[i].label = -1;
    }
    __syncwarp();

    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
        phelper1 = ld.vertices[i].vertexid;
        phelper2 = d_bsearch_array(dd.twohop_neighbors + dd.twohop_offsets[pvertexid], dd.twohop_offsets[pvertexid + 1] - dd.twohop_offsets[pvertexid], phelper1);

        if (phelper2 > -1) {
            ld.vertices[i].label = 0;
            dd.lane_candidate_indegs[lane_write + lane_remaining_count++] = ld.vertices[i].indeg;
        }
    }
    __syncwarp();

    // scan to calculate write postion in warp arrays
    phelper2 = lane_remaining_count;
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
        if (LANE_IDX >= i) {
            lane_remaining_count += phelper1;
        }
        __syncwarp();
    }
    // lane remaining count sum is scan for last lane and its value
    if (LANE_IDX == WARP_SIZE - 1) {
        wd.remaining_count[WIB_IDX] = lane_remaining_count;
    }
    // make scan exclusive
    lane_remaining_count -= phelper2;
    __syncwarp();

    // parallel write lane arrays to warp array
    for (int i = 0; i < phelper2; i++) {
        dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + lane_remaining_count + i] = dd.lane_candidate_indegs[lane_write + i];
    }
    __syncwarp();
}

__device__ void d_diameter_pruning_cv(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int number_of_crit_adj)
{
    // (WVERTICES_SIZE * WARP_IDX) /warp write location to adjacencies
    // vertices size * warp idx + (vertices size / warp size) * lane idx
    int lane_write = ((WVERTICES_SIZE * WARP_IDX) + ((WVERTICES_SIZE / WARP_SIZE) * LANE_IDX));

    // vertex iteration
    int lane_remaining_count;

    // intersection
    int phelper1;
    int phelper2;



    lane_remaining_count = 0;

    // remove all cands who are not within 2hops of all newly added cands
    for (int k = wd.number_of_members[WIB_IDX] + LANE_IDX; k < wd.total_vertices[WIB_IDX]; k += WARP_SIZE) {
        if (dd.adjacencies[(WVERTICES_SIZE * WARP_IDX) + k] == number_of_crit_adj) {
            dd.lane_candidate_indegs[lane_write + lane_remaining_count++] = ld.vertices[k].indeg;
        }
        else {
            ld.vertices[k].label = -1;
        }
    }

    // scan to calculate write postion in warp arrays
    phelper2 = lane_remaining_count;
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
        if (LANE_IDX >= i) {
            lane_remaining_count += phelper1;
        }
        __syncwarp();
    }
    // lane remaining count sum is scan for last lane and its value
    if (LANE_IDX == WARP_SIZE - 1) {
        wd.remaining_count[WIB_IDX] = lane_remaining_count;
    }
    // make scan exclusive
    lane_remaining_count -= phelper2;
    __syncwarp();

    // parallel write lane arrays to warp array
    for (int i = 0; i < phelper2; i++) {
        dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + lane_remaining_count + i] = dd.lane_candidate_indegs[lane_write + i];
    }
    __syncwarp();
}

// returns true if invalid bounds or failed found
__device__ bool d_degree_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    // vertices size * warp idx + (vertices size / warp size) * lane idx
    int lane_write = ((WVERTICES_SIZE * WARP_IDX) + ((WVERTICES_SIZE / WARP_SIZE) * LANE_IDX));

    // helper variables used throughout method to store various values, names have no meaning
    int pvertexid;
    int phelper1;
    int phelper2;
    Vertex* read;
    Vertex* write;

    // counter for lane intersection results
    int lane_remaining_count;
    int lane_removed_count;



    d_sort_i(dd.candidate_indegs + (WVERTICES_SIZE * WARP_IDX), wd.remaining_count[WIB_IDX], d_sort_degs);

    d_calculate_LU_bounds(dd, wd, ld, wd.remaining_count[WIB_IDX]);
    if (wd.invalid_bounds[WIB_IDX]) {
        return true;
    }

    // check for failed vertices
    if (LANE_IDX == 0) {
        wd.success[WIB_IDX] = false;
    }
    __syncwarp();
    for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX] && !wd.success[WIB_IDX]; k += WARP_SIZE) {
        if (!d_vert_isextendable_LU(ld.vertices[k], dd, wd, ld)) {
            wd.success[WIB_IDX] = true;
            break;
        }

    }
    __syncwarp();
    if (wd.success[WIB_IDX]) {
        return true;
    }

    if (LANE_IDX == 0) {
        wd.remaining_count[WIB_IDX] = 0;
        wd.removed_count[WIB_IDX] = 0;
        wd.rw_counter[WIB_IDX] = 0;
    }



    lane_remaining_count = 0;
    lane_removed_count = 0;
    
    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
        if (ld.vertices[i].label == 0 && d_cand_isvalid_LU(ld.vertices[i], dd, wd, ld)) {
            dd.lane_remaining_candidates[lane_write + lane_remaining_count++] = i;
        }
        else {
            dd.lane_removed_candidates[lane_write + lane_removed_count++] = i;
        }
    }
    __syncwarp();

    // scan to calculate write postion in warp arrays
    phelper2 = lane_remaining_count;
    pvertexid = lane_removed_count;
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
        if (LANE_IDX >= i) {
            lane_remaining_count += phelper1;
        }
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_removed_count, i, WARP_SIZE);
        if (LANE_IDX >= i) {
            lane_removed_count += phelper1;
        }
        __syncwarp();
    }
    // lane remaining count sum is scan for last lane and its value
    if (LANE_IDX == WARP_SIZE - 1) {
        wd.remaining_count[WIB_IDX] = lane_remaining_count;
        wd.removed_count[WIB_IDX] = lane_removed_count;
    }
    // make scan exclusive
    lane_remaining_count -= phelper2;
    lane_removed_count -= pvertexid;

    // parallel write lane arrays to warp array
    for (int i = 0; i < phelper2; i++) {
        dd.remaining_candidates[(WVERTICES_SIZE * WARP_IDX) + lane_remaining_count + i] = ld.vertices[dd.lane_remaining_candidates[lane_write + i]];
    }
    // only need removed if going to be using removed to update degrees
    if (!(wd.remaining_count[WIB_IDX] < wd.removed_count[WIB_IDX])) {
        for (int i = 0; i < pvertexid; i++) {
            dd.removed_candidates[(WVERTICES_SIZE * WARP_IDX) + lane_removed_count + i] = ld.vertices[dd.lane_removed_candidates[lane_write + i]].vertexid;
        }
    }
    __syncwarp();


    
    while (wd.remaining_count[WIB_IDX] > 0 && wd.removed_count[WIB_IDX] > 0) {
        // different blocks for the read and write locations, vertices and remaining, this is done to avoid using extra variables and only one condition
        if (wd.rw_counter[WIB_IDX] % 2 == 0) {
            read = dd.remaining_candidates + (WVERTICES_SIZE * WARP_IDX);
            write = ld.vertices + wd.number_of_members[WIB_IDX];
        }
        else {
            read = ld.vertices + wd.number_of_members[WIB_IDX];
            write = dd.remaining_candidates + (WVERTICES_SIZE * WARP_IDX);
        }

        // update degrees
        if (wd.remaining_count[WIB_IDX] < wd.removed_count[WIB_IDX]) {
            // via remaining, reset exdegs
            for (int i = LANE_IDX; i < wd.number_of_members[WIB_IDX]; i += WARP_SIZE) {
                ld.vertices[i].exdeg = 0;
            }
            for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
                read[i].exdeg = 0;
            }
            __syncwarp();



            // update exdeg based on remaining candidates, every lane should get the next vertex to intersect dynamically
            for (int i = LANE_IDX; i < wd.number_of_members[WIB_IDX]; i += WARP_SIZE) {
                pvertexid = ld.vertices[i].vertexid;

                for (int j = 0; j < wd.remaining_count[WIB_IDX]; j++) {
                    phelper1 = read[j].vertexid;
                    phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                    if (phelper2 > -1) {
                        ld.vertices[i].exdeg++;
                    }
                }
            }

            for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
                pvertexid = read[i].vertexid;

                for (int j = 0; j < wd.remaining_count[WIB_IDX]; j++) {
                    if (j == i) {
                        continue;
                    }

                    phelper1 = read[j].vertexid;
                    phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                    if (phelper2 > -1) {
                        read[i].exdeg++;
                    }
                }
            }
        }
        else {
            // via removed, update exdeg based on remaining candidates, again lane scheduling should be dynamic
            for (int i = LANE_IDX; i < wd.number_of_members[WIB_IDX]; i += WARP_SIZE) {
                pvertexid = ld.vertices[i].vertexid;

                for (int j = 0; j < wd.removed_count[WIB_IDX]; j++) {
                    phelper1 = dd.removed_candidates[(WVERTICES_SIZE * WARP_IDX) + j];
                    phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                    if (phelper2 > -1) {
                        ld.vertices[i].exdeg--;
                    }
                }
            }

            for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
                pvertexid = read[i].vertexid;

                for (int j = 0; j < wd.removed_count[WIB_IDX]; j++) {
                    phelper1 = dd.removed_candidates[(WVERTICES_SIZE * WARP_IDX) + j];
                    phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                    if (phelper2 > -1) {
                        read[i].exdeg--;
                    }
                }
            }
        }
        __syncwarp();

        lane_remaining_count = 0;

        for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
            if (d_cand_isvalid_LU(read[i], dd, wd, ld)) {
                dd.lane_candidate_indegs[lane_write + lane_remaining_count++] = read[i].indeg;
            }
        }
        __syncwarp();

        // scan to calculate write postion in warp arrays
        phelper2 = lane_remaining_count;
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
            if (LANE_IDX >= i) {
                lane_remaining_count += phelper1;
            }
            __syncwarp();
        }
        // lane remaining count sum is scan for last lane and its value
        if (LANE_IDX == WARP_SIZE - 1) {
            wd.num_val_cands[WIB_IDX] = lane_remaining_count;
        }
        // make scan exclusive
        lane_remaining_count -= phelper2;

        // parallel write lane arrays to warp array
        for (int i = 0; i < phelper2; i++) {
            dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + lane_remaining_count + i] = dd.lane_candidate_indegs[lane_write + i];
        }
        __syncwarp();



        d_sort_i(dd.candidate_indegs + (WVERTICES_SIZE * WARP_IDX), wd.num_val_cands[WIB_IDX], d_sort_degs);

        d_calculate_LU_bounds(dd, wd, ld, wd.num_val_cands[WIB_IDX]);
        if (wd.invalid_bounds[WIB_IDX]) {
            return true;
        }

        // check for failed vertices
        for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX] && !wd.success[WIB_IDX]; k += WARP_SIZE) {
            if (!d_vert_isextendable_LU(ld.vertices[k], dd, wd, ld)) {
                wd.success[WIB_IDX] = true;
                break;
            }

        }
        __syncwarp();
        if (wd.success[WIB_IDX]) {
            return true;
        }



        lane_remaining_count = 0;
        lane_removed_count = 0;

        // check for failed candidates
        for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
            if (d_cand_isvalid_LU(read[i], dd, wd, ld)) {
                dd.lane_remaining_candidates[lane_write + lane_remaining_count++] = i;
            }
            else {
                dd.lane_removed_candidates[lane_write + lane_removed_count++] = i;
            }
        }
        __syncwarp();

        // scan to calculate write postion in warp arrays
        phelper2 = lane_remaining_count;
        pvertexid = lane_removed_count;
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
            if (LANE_IDX >= i) {
                lane_remaining_count += phelper1;
            }
            phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_removed_count, i, WARP_SIZE);
            if (LANE_IDX >= i) {
                lane_removed_count += phelper1;
            }
            __syncwarp();
        }
        // lane remaining count sum is scan for last lane and its value
        if (LANE_IDX == WARP_SIZE - 1) {
            wd.num_val_cands[WIB_IDX] = lane_remaining_count;
            wd.removed_count[WIB_IDX] = lane_removed_count;
        }
        // make scan exclusive
        lane_remaining_count -= phelper2;
        lane_removed_count -= pvertexid;

        // parallel write lane arrays to warp array
        for (int i = 0; i < phelper2; i++) {
            write[lane_remaining_count + i] = read[dd.lane_remaining_candidates[lane_write + i]];
        }
        // only need removed if going to be using removed to update degrees
        if (!(wd.num_val_cands[WIB_IDX] < wd.removed_count[WIB_IDX])) {
            for (int i = 0; i < pvertexid; i++) {
                dd.removed_candidates[(WVERTICES_SIZE * WARP_IDX) + lane_removed_count + i] = read[dd.lane_removed_candidates[lane_write + i]].vertexid;
            }
        }



        if (LANE_IDX == 0) {
            wd.remaining_count[WIB_IDX] = wd.num_val_cands[WIB_IDX];
            wd.rw_counter[WIB_IDX]++;
        }
    }



    // condense vertices so remaining are after members, only needs to be done if they were not written into vertices last time
    if (wd.rw_counter[WIB_IDX] % 2 == 0) {
        for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
            ld.vertices[wd.number_of_members[WIB_IDX] + i] = dd.remaining_candidates[(WVERTICES_SIZE * WARP_IDX) + i];
        }
    }

    if (LANE_IDX == 0) {
        wd.total_vertices[WIB_IDX] = wd.total_vertices[WIB_IDX] - wd.number_of_candidates[WIB_IDX] + wd.remaining_count[WIB_IDX];
        wd.number_of_candidates[WIB_IDX] = wd.remaining_count[WIB_IDX];
    }

    return false;
}

__device__ void d_calculate_LU_bounds(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int number_of_candidates)
{
    int index;

    int min_clq_indeg;
    int min_indeg_exdeg;
    int min_clq_totaldeg;
    int sum_clq_indeg;

    // initialize the values of the LU calculation variables to the first vertices values so they can be compared to other vertices without error
    min_clq_indeg = ld.vertices[0].indeg;
    min_indeg_exdeg = ld.vertices[0].exdeg;
    min_clq_totaldeg = ld.vertices[0].indeg + ld.vertices[0].exdeg;
    sum_clq_indeg = 0;

    // each warp also has a copy of these variables to allow for intra-warp comparison of these variables.
    if (LANE_IDX == 0) {
        wd.invalid_bounds[WIB_IDX] = false;

        wd.sum_candidate_indeg[WIB_IDX] = 0;
        wd.tightened_upper_bound[WIB_IDX] = 0;

        wd.min_clq_indeg[WIB_IDX] = ld.vertices[0].indeg;
        wd.min_indeg_exdeg[WIB_IDX] = ld.vertices[0].exdeg;
        wd.min_clq_totaldeg[WIB_IDX] = ld.vertices[0].indeg + ld.vertices[0].exdeg;
        wd.sum_clq_indeg[WIB_IDX] = ld.vertices[0].indeg;

        wd.min_ext_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX] + 1, dd);
    }
    __syncwarp();

    // each warp finds these values on their subsection of vertices
    for (index = 1 + LANE_IDX; index < wd.number_of_members[WIB_IDX]; index += WARP_SIZE) {
        sum_clq_indeg += ld.vertices[index].indeg;

        if (ld.vertices[index].indeg < min_clq_indeg) {
            min_clq_indeg = ld.vertices[index].indeg;
            min_indeg_exdeg = ld.vertices[index].exdeg;
        }
        else if (ld.vertices[index].indeg == min_clq_indeg) {
            if (ld.vertices[index].exdeg < min_indeg_exdeg) {
                min_indeg_exdeg = ld.vertices[index].exdeg;
            }
        }

        if (ld.vertices[index].indeg + ld.vertices[index].exdeg < min_clq_totaldeg) {
            min_clq_totaldeg = ld.vertices[index].indeg + ld.vertices[index].exdeg;
        }
    }

    // get sum
    for (int i = 1; i < 32; i *= 2) {
        sum_clq_indeg += __shfl_xor_sync(0xFFFFFFFF, sum_clq_indeg, i);
    }
    if (LANE_IDX == 0) {
        // add to shared memory sum
        wd.sum_clq_indeg[WIB_IDX] += sum_clq_indeg;
    }
    __syncwarp();

    // CRITICAL SECTION - each lane then compares their values to the next to get a warp level value
    for (int i = 0; i < WARP_SIZE; i++) {
        if (LANE_IDX == i) {
            if (min_clq_indeg < wd.min_clq_indeg[WIB_IDX]) {
                wd.min_clq_indeg[WIB_IDX] = min_clq_indeg;
                wd.min_indeg_exdeg[WIB_IDX] = min_indeg_exdeg;
            }
            else if (min_clq_indeg == wd.min_clq_indeg[WIB_IDX]) {
                if (min_indeg_exdeg < wd.min_indeg_exdeg[WIB_IDX]) {
                    wd.min_indeg_exdeg[WIB_IDX] = min_indeg_exdeg;
                }
            }

            if (min_clq_totaldeg < wd.min_clq_totaldeg[WIB_IDX]) {
                wd.min_clq_totaldeg[WIB_IDX] = min_clq_totaldeg;
            }
        }
        __syncwarp();
    }

    if (LANE_IDX == 0) {
        if (wd.min_clq_indeg[WIB_IDX] < dd.minimum_degrees[wd.number_of_members[WIB_IDX]])
        {
            // lower
            wd.lower_bound[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX], dd) - min_clq_indeg;

            while (wd.lower_bound[WIB_IDX] <= wd.min_indeg_exdeg[WIB_IDX] && wd.min_clq_indeg[WIB_IDX] + wd.lower_bound[WIB_IDX] <
                dd.minimum_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]]) {
                wd.lower_bound[WIB_IDX]++;
            }

            if (wd.min_clq_indeg[WIB_IDX] + wd.lower_bound[WIB_IDX] < dd.minimum_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]]) {
                wd.invalid_bounds[WIB_IDX] = true;
            }

            // upper
            wd.upper_bound[WIB_IDX] = floor(wd.min_clq_totaldeg[WIB_IDX] / (*(dd.minimum_degree_ratio))) + 1 - wd.number_of_members[WIB_IDX];

            if (wd.upper_bound[WIB_IDX] > number_of_candidates) {
                wd.upper_bound[WIB_IDX] = number_of_candidates;
            }

            // tighten
            if (wd.lower_bound[WIB_IDX] < wd.upper_bound[WIB_IDX]) {
                // tighten lower
                for (index = 0; index < wd.lower_bound[WIB_IDX]; index++) {
                    wd.sum_candidate_indeg[WIB_IDX] += dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + index];
                }

                while (index < wd.upper_bound[WIB_IDX] && wd.sum_clq_indeg[WIB_IDX] + wd.sum_candidate_indeg[WIB_IDX] < wd.number_of_members[WIB_IDX] *
                    dd.minimum_degrees[wd.number_of_members[WIB_IDX] + index]) {
                    wd.sum_candidate_indeg[WIB_IDX] += dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + index];
                    index++;
                }

                if (wd.sum_clq_indeg[WIB_IDX] + wd.sum_candidate_indeg[WIB_IDX] < wd.number_of_members[WIB_IDX] * dd.minimum_degrees[wd.number_of_members[WIB_IDX] + index]) {
                    wd.invalid_bounds[WIB_IDX] = true;
                }
                else {
                    wd.lower_bound[WIB_IDX] = index;

                    wd.tightened_upper_bound[WIB_IDX] = index;

                    while (index < wd.upper_bound[WIB_IDX]) {
                        wd.sum_candidate_indeg[WIB_IDX] += dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + index];

                        index++;

                        if (wd.sum_clq_indeg[WIB_IDX] + wd.sum_candidate_indeg[WIB_IDX] >= wd.number_of_members[WIB_IDX] *
                            dd.minimum_degrees[wd.number_of_members[WIB_IDX] + index]) {
                            wd.tightened_upper_bound[WIB_IDX] = index;
                        }
                    }

                    if (wd.upper_bound[WIB_IDX] > wd.tightened_upper_bound[WIB_IDX]) {
                        wd.upper_bound[WIB_IDX] = wd.tightened_upper_bound[WIB_IDX];
                    }

                    if (wd.lower_bound[WIB_IDX] > 1) {
                        wd.min_ext_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX], dd);
                    }
                }
            }
        }
        else {
            wd.min_ext_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX] + 1,
                dd);

            wd.upper_bound[WIB_IDX] = number_of_candidates;

            if (wd.number_of_members[WIB_IDX] < (*(dd.minimum_clique_size))) {
                wd.lower_bound[WIB_IDX] = (*(dd.minimum_clique_size)) - wd.number_of_members[WIB_IDX];
            }
            else {
                wd.lower_bound[WIB_IDX] = 0;
            }
        }

        if (wd.number_of_members[WIB_IDX] + wd.upper_bound[WIB_IDX] < (*(dd.minimum_clique_size))) {
            wd.invalid_bounds[WIB_IDX] = true;
        }

        if (wd.upper_bound[WIB_IDX] < 0 || wd.upper_bound[WIB_IDX] < wd.lower_bound[WIB_IDX]) {
            wd.invalid_bounds[WIB_IDX] = true;
        }
    }
    __syncwarp();
}

__device__ void d_check_for_clique(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    bool clique = true;

    for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX]; k += WARP_SIZE) {
        if (ld.vertices[k].indeg < dd.minimum_degrees[wd.number_of_members[WIB_IDX]]) {
            clique = false;
            break;
        }
    }
    // set to false if any threads in warp do not meet degree requirement
    clique = !(__any_sync(0xFFFFFFFF, !clique));

    // if clique write to warp buffer for cliques
    if (clique) {
        uint64_t start_write = (WCLIQUES_SIZE * WARP_IDX) + dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * WARP_IDX) + (dd.wcliques_count[WARP_IDX])];
        for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX]; k += WARP_SIZE) {
            dd.wcliques_vertex[start_write + k] = ld.vertices[k].vertexid;
        }
        if (LANE_IDX == 0) {
            (dd.wcliques_count[WARP_IDX])++;
            dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * WARP_IDX) + (dd.wcliques_count[WARP_IDX])] = start_write - (WCLIQUES_SIZE * WARP_IDX) + wd.number_of_members[WIB_IDX];
        }
    }
}

__device__ void d_write_to_tasks(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    uint64_t start_write = (WTASKS_SIZE * WARP_IDX) + dd.wtasks_offset[WTASKS_OFFSET_SIZE * WARP_IDX + (dd.wtasks_count[WARP_IDX])];

    for (int k = LANE_IDX; k < wd.total_vertices[WIB_IDX]; k += WARP_SIZE) {
        dd.wtasks_vertices[start_write + k].vertexid = ld.vertices[k].vertexid;
        dd.wtasks_vertices[start_write + k].label = ld.vertices[k].label;
        dd.wtasks_vertices[start_write + k].indeg = ld.vertices[k].indeg;
        dd.wtasks_vertices[start_write + k].exdeg = ld.vertices[k].exdeg;
        dd.wtasks_vertices[start_write + k].lvl2adj = 0;
    }
    if (LANE_IDX == 0) {
        (dd.wtasks_count[WARP_IDX])++;
        dd.wtasks_offset[(WTASKS_OFFSET_SIZE * WARP_IDX) + (dd.wtasks_count[WARP_IDX])] = start_write - (WTASKS_SIZE * WARP_IDX) + wd.total_vertices[WIB_IDX];
    }
}