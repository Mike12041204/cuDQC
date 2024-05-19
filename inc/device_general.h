#ifndef DCUQC_DEVICE_GENERAL_H
#define DCUQC_DEVICE_GENERAL_H

#include "./common.h"

__global__ void d_expand_level(GPU_Data dd);
__global__ void transfer_buffers(GPU_Data dd);
__global__ void fill_from_buffer(GPU_Data dd);

#endif // DCUQC_DEVICE_GENERAL_H