// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#define _4HALF2_            4
#define _8HALF_             8
#define _INT4_TO_4INT_      4
#define _INT4_TO_8HALF_     8

#define HMAX2_INST(_d, _a, _b, _c) \
        asm volatile("vmax2.s32.s32.s32 %0, %1, %2, %3;\n":   "=r"(_d): "r"(_a), "r"(_b), "r"(_c));

#define HMIN2_INST(_d, _a, _b, _c) \
        asm volatile("vmin2.s32.s32.s32 %0, %1, %2, %3;\n":   "=r"(_d): "r"(_a), "r"(_b), "r"(_c));

#include <cuda_fp16.h>

__global__ void MergeConvSplitResults(
        int4* input,             int4* output, 
	    int split_height_v1,     int split_width_v8, 
	    int out_hw,              int split, 
        int has_bias,            const int4* bias,
        int  has_relu,           const int clip_min,
        bool has_clip,           const int clip_max,
        bool has_elt,            const int4* pre_data,
        int  has_elt_relu,       const int elt_clip_min,
        bool has_elt_clip,       const int elt_clip_max,
        bool has_concat,         int concat_offset_v8,
        int concat_stride_v8)
{
#if (__CUDA_ARCH__ >= 600) && (__CUDACC_VER_MAJOR__ >= 9)
    int k_id = blockIdx.y * blockDim.x + threadIdx.x;
    int64_t nhw_id = blockIdx.x;

    int off  = nhw_id * split_width_v8 + k_id;

    const int4 ZEROv4 = {0, 0, 0, 0};
    bool is_in_range = k_id < split_width_v8;

    int4 merge_v4, split_v4, bias_v4;

    __half2 * h2_merge = (__half2 *) &merge_v4;
    __half2 * h2_split = (__half2 *) &split_v4;
    __half2 * h2_bias  = (__half2 *) &bias_v4;

    merge_v4 = is_in_range ? input[off] : ZEROv4;

    for(int i = 1; i < split; i++)
    {
        split_v4 = is_in_range ? input[off + i * split_height_v1 * split_width_v8] : ZEROv4;

	    for(int j = 0; j < _4HALF2_; j++)
	        h2_merge[j] = __hadd2(h2_merge[j], h2_split[j]);
    }
    if(has_bias)
    {
        bias_v4 = is_in_range ? ((int4 *) bias) [k_id] : ZEROv4;

#pragma unroll
	    for(int j = 0; j < _4HALF2_; j++)
	        h2_merge[j] = __hadd2(h2_merge[j], h2_bias[j]);
    }

    int *    merge_v1  = (int *)    &merge_v4;

    if(has_relu)
    {
        for(int i = 0; i < _4HALF2_; i++)
            merge_v1[i] = __vmaxs2(merge_v1[i], 0);
    }

    if(has_clip) {
#pragma unroll
        for(int i = 0; i < _4HALF2_; i++)
        {
            HMIN2_INST(merge_v1[i], merge_v1[i], clip_max, merge_v1[i]); \
            HMAX2_INST(merge_v1[i], merge_v1[i], clip_min, merge_v1[i]); \
        }
    }

    if(has_elt) {
	    int4 eltV4     = is_in_range ? pre_data[off] : ZEROv4;
	    __half2* h2Elt = (__half2*) &eltV4;

	    for(int i = 0; i < _INT4_TO_4INT_; i++)
	        h2_merge[i] = __hadd2(h2_merge[i], h2Elt[i]);
    }

    if(has_elt_relu) {
        for(int i = 0; i < _4HALF2_; i++)
            merge_v1[i] = __vmaxs2(merge_v1[i], 0);
    }

    if(has_elt_clip) {
        for(int i = 0; i < _4HALF2_; i++) {
            HMIN2_INST(merge_v1[i], merge_v1[i], elt_clip_max, merge_v1[i]); \
            HMAX2_INST(merge_v1[i], merge_v1[i], elt_clip_min, merge_v1[i]); \
        }
    }

    int concat_v8_off = 0;
    if(has_concat){
	    concat_v8_off = concat_offset_v8 + nhw_id * concat_stride_v8;
	    off = concat_v8_off + k_id;
    }
    
    if(is_in_range) output[off] = merge_v4;

#endif
}
