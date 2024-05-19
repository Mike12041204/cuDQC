#ifndef DCUQC_HOST_HELPER_H
#define DCUQC_HOST_HELPER_H

#include "./common.h"

int h_sort_vert_cv(const void* a, const void* b);
int h_sort_vert_Q(const void* a, const void* b);
int h_sort_desc(const void* a, const void* b);
inline int h_get_mindeg(int clique_size);
inline bool h_cand_isvalid_LU(Vertex vertex, int clique_size, int upper_bound, int lower_bound, int min_ext_deg);
inline bool  h_vert_isextendable_LU(Vertex vertex, int clique_size, int upper_bound, int lower_bound, int min_ext_deg);

#endif // DCUQC_HOST_HELPER_H