#include "../inc/common.h"
#include "../inc/device_helper.h"
#include "../inc/device_debug.h"

// searches an int array for a certain int, returns the position in the array that item was found, or -1 if not found
__device__ int d_bsearch_array(int* search_array, int array_size, int search_number)
{
    // ALGO - binary
    // TYPE - serial
    // SPEED - 0(log(n))

    int low = 0;
    int high = array_size - 1;

    while (low <= high) {
        int mid = (low + high) / 2;

        if (search_array[mid] == search_number) {
            return mid;
        }
        else if (search_array[mid] > search_number) {
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }

    return -1;
}

// consider using merge
__device__ void d_sort(Vertex* target, int size, int (*func)(Vertex&, Vertex&))
{
    // ALGO - ODD/EVEN
    // TYPE - PARALLEL
    // SPEED - O(n^2)

    Vertex vertex1;
    Vertex vertex2;

    for (int i = 0; i < size; i++) {
        for (int j = (i % 2) + (LANE_IDX * 2); j < size - 1; j += (WARP_SIZE * 2)) {
            vertex1 = target[j];
            vertex2 = target[j + 1];

            if (func(vertex1, vertex2) == 1) {
                target[j] = vertex2;
                target[j + 1] = vertex1;
            }
        }
        __syncwarp();
    }
}

__device__ void d_sort_i(int* target, int size, int (*func)(int, int))
{
    // ALGO - ODD/EVEN
    // TYPE - PARALLEL
    // SPEED - O(n^2)

    int num1;
    int num2;

    for (int i = 0; i < size; i++) {
        for (int j = (i % 2) + (LANE_IDX * 2); j < size - 1; j += (WARP_SIZE * 2)) {
            num1 = target[j];
            num2 = target[j + 1];

            if (func(num1, num2) == 1) {
                target[j] = num2;
                target[j + 1] = num1;
            }
        }
        __syncwarp();
    }
}

// Quick enumeration order sort keys
__device__ int d_sort_vert_Q(Vertex& v1, Vertex& v2)
{
    // order is: member -> covered -> cands -> cover
    // keys are: indeg -> exdeg -> lvl2adj -> vertexid

    if (v1.label == 1 && v2.label != 1)
        return -1;
    else if (v1.label != 1 && v2.label == 1)
        return 1;
    else if (v1.label == 2 && v2.label != 2)
        return -1;
    else if (v1.label != 2 && v2.label == 2)
        return 1;
    else if (v1.label == 0 && v2.label != 0)
        return -1;
    else if (v1.label != 0 && v2.label == 0)
        return 1;
    else if (v1.label == 3 && v2.label != 3)
        return -1;
    else if (v1.label != 3 && v2.label == 3)
        return 1;
    else if (v1.indeg > v2.indeg)
        return -1;
    else if (v1.indeg < v2.indeg)
        return 1;
    else if (v1.exdeg > v2.exdeg)
        return -1;
    else if (v1.exdeg < v2.exdeg)
        return 1;
    else if (v1.lvl2adj > v2.lvl2adj)
        return -1;
    else if (v1.lvl2adj < v2.lvl2adj)
        return 1;
    else if (v1.vertexid > v2.vertexid)
        return -1;
    else if (v1.vertexid < v2.vertexid)
        return 1;
    else
        return 0;
}

__device__ int d_sort_vert_cv(Vertex& v1, Vertex& v2)
{
    // put crit adj vertices before candidates

    if (v1.label == 4 && v2.label != 4)
        return -1;
    else if (v1.label != 4 && v2.label == 4)
        return 1;
    else
        return 0;
}

__device__ int d_sort_degs(int n1, int n2)
{
    // descending order

    if (n1 > n2) {
        return -1;
    }
    else if (n1 < n2) {
        return 1;
    }
    else {
        return 0;
    }
}

__device__ int d_get_mindeg(int number_of_members, GPU_Data& dd)
{
    if (number_of_members < (*(dd.minimum_clique_size))) {
        return dd.minimum_degrees[(*(dd.minimum_clique_size))];
    }
    else {
        return dd.minimum_degrees[number_of_members];
    }
}

__device__ bool d_cand_isvalid_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    if (vertex.indeg + vertex.exdeg < dd.minimum_degrees[(*(dd.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + vertex.exdeg + 1, dd)) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < wd.min_ext_deg[WIB_IDX]) {
        return false;
    }
    else if (vertex.indeg + wd.upper_bound[WIB_IDX] - 1 < dd.minimum_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX], dd)) {
        return false;
    }
    else {
        return true;
    }
}

__device__ bool d_vert_isextendable_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    if (vertex.indeg + vertex.exdeg < dd.minimum_degrees[(*(dd.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + vertex.exdeg, dd)) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < wd.min_ext_deg[WIB_IDX]) {
        return false;
    }
    else if (vertex.exdeg == 0 && vertex.indeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + vertex.exdeg, dd)) {
        return false;
    }
    else if (vertex.indeg + wd.upper_bound[WIB_IDX] < dd.minimum_degrees[wd.number_of_members[WIB_IDX] + wd.upper_bound[WIB_IDX]]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX], dd)) {
        return false;
    }
    else {
        return true;
    }
}