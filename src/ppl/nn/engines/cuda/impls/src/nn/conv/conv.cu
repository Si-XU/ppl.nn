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

#include <fstream>
#include <iostream>
using namespace std;

#include <vector>
#include <cuda.h>
#include <assert.h>

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "cudakernel/nn/conv/conv_fp16.h"
#include "kernel_type.h"
#include "conv_common.h"
#include "common/init_lut.h"
#include "common/merge_split.h"

#include "float.h"

#define TIMES 4

#define SPK_KPARAM_LIST \
            pad_input,                                                                  \
            d_flt,                                                                      \
            conv_out,                                                                   \
            kloop_num,                                                                  \
    		in_lut,                        in_lut_size,                                 \
    		flt_lut,                       flt_lut_size,                                \
            num_chl_per_spk_head,          num_chl_per_spk_tail,                        \
            in_hw,                         out_hw,                                      \
            flt_hw,                        splitk,                                      \
            conv_param.in_height,          conv_param.in_width,                         \
            conv_param.in_num,             conv_param.num_grp,                          \
            num_chl_per_grp,               num_chl_per_grp_pad,                         \
            conv_param.flt_height,         conv_param.flt_width,                        \
            num_flt_per_grp,               num_flt_per_grp_pad,                         \
            conv_param.out_height,         conv_param.out_width,                        \
            conv_param.stride_height,      conv_param.stride_width,                     \
            conv_param.pad_height,         conv_param.pad_width,                        \
            conv_param.hole_height,        conv_param.hole_width,                       \
            conv_param.has_bias,           (int *)bias

#define LUT_KPARAM_LIST \
            pad_input,                                                                  \
            d_flt,                                                                      \
            conv_out,                                                                   \
            kloop_num,                                                                  \
    		in_lut,                        in_lut_size,                                 \
    		flt_lut,                       flt_lut_size,                                \
            in_hw,                         out_hw,                                      \
            flt_hw,                        splitk,                                      \
            conv_param.in_height,          conv_param.in_width,                         \
            conv_param.in_num,             conv_param.num_grp,                          \
            num_chl_per_grp,               num_chl_per_grp_pad,                         \
            conv_param.flt_height,         conv_param.flt_width,                        \
            num_flt_per_grp,               num_flt_per_grp_pad,                         \
            conv_param.out_height,         conv_param.out_width,                        \
            conv_param.stride_height,      conv_param.stride_width,                     \
            conv_param.pad_height,         conv_param.pad_width,                        \
            conv_param.hole_height,        conv_param.hole_width,                       \
            conv_param.has_bias,           bias,                                        \
            fuse_param.has_activation,     clip_min,                                    \
            fuse_param.has_clip,           clip_max,                                    \
            fuse_param.has_elt,            (const int4 *) fuse_param.pre_data,          \
            fuse_param.has_elt_activation, elt_clip_min,                                \
            fuse_param.has_elt_clip,       elt_clip_max,                                \
            fuse_param.has_concat,         concat_offset_v8,                            \
            concat_stride_v8

#define SWZL_SPK_KPARAM_LIST \
            d_flt,                                                                      \
            pad_input,                                                                  \
            conv_out,                                                                   \
            kloop_num,                                                                  \
            in_lut,                        in_lut_size,                                 \
            flt_lut,                       flt_lut_size,                                \
            num_chl_per_spk_head,          num_chl_per_spk_tail,                        \
            in_hw,                         out_hw,                                      \
            flt_hw,                        splitk,                                      \
            conv_param.in_height,          conv_param.in_width,                         \
            conv_param.in_num,             conv_param.num_grp,                          \
            num_chl_per_grp,               num_chl_per_grp_pad,                         \
            conv_param.flt_height,         conv_param.flt_width,                        \
            num_flt_per_grp,               num_flt_per_grp_pad,                         \
            conv_param.out_height,         conv_param.out_width,                        \
            conv_param.stride_height,      conv_param.stride_width,                     \
            conv_param.pad_height,         conv_param.pad_width,                        \
            conv_param.hole_height,        conv_param.hole_width,                       \
            conv_param.has_bias,           (int *)bias

#define SWZL_LUT_KPARAM_LIST \
            d_flt,                                                                      \
            pad_input,                                                                  \
            conv_out,                                                                   \
            kloop_num,                                                                  \
            in_lut,                        in_lut_size,                                 \
            flt_lut,                       flt_lut_size,                                \
            in_hw,                         out_hw,                                      \
            flt_hw,                        splitk,                                      \
            conv_param.in_height,          conv_param.in_width,                         \
            conv_param.in_num,             conv_param.num_grp,                          \
            num_chl_per_grp,               num_chl_per_grp_pad,                         \
            conv_param.flt_height,         conv_param.flt_width,                        \
            num_flt_per_grp,               num_flt_per_grp_pad,                         \
            conv_param.out_height,         conv_param.out_width,                        \
            conv_param.stride_height,      conv_param.stride_width,                     \
            conv_param.pad_height,         conv_param.pad_width,                        \
            conv_param.hole_height,        conv_param.hole_width,                       \
            conv_param.has_bias,           bias,                                        \
            fuse_param.has_activation,     clip_min,                                    \
            fuse_param.has_clip,           clip_max,                                    \
            fuse_param.has_elt,            (const int4 *) fuse_param.pre_data,          \
            fuse_param.has_elt_activation, elt_clip_min,                                \
            fuse_param.has_elt_clip,       elt_clip_max,                                \
            fuse_param.has_concat,         concat_offset_v8,                            \
            concat_stride_v8

#define IDX_KPARAM_LIST \
            pad_input,                                                                  \
            d_flt,                                                                      \
            conv_out,                                                                   \
            kloop_num,                      koff_num_pad,                               \
            in_hw,                         out_hw,                                      \
            flt_hw,                        out_nhw,                                     \
            conv_param.in_height,          conv_param.in_width,                         \
            conv_param.in_num,             conv_param.num_grp,                          \
            conv_param.num_chl,            num_chl_per_grp,                             \
            in_chl_per_grp_pad,            flt_chl_per_grp_pad,                         \
            conv_param.flt_height,         conv_param.flt_width,                        \
            num_flt_per_grp,               num_flt_per_grp_pad,                         \
            conv_param.out_height,         conv_param.out_width,                        \
            conv_param.stride_height,      conv_param.stride_width,                     \
            conv_param.pad_height,         conv_param.pad_width,                        \
            conv_param.hole_height,        conv_param.hole_width,                       \
            conv_param.has_bias,           bias,                                        \
            fuse_param.has_activation,     clip_min,                                    \
            fuse_param.has_clip,           clip_max,                                    \
            fuse_param.has_elt,            (const int4 *) fuse_param.pre_data,          \
            fuse_param.has_elt_activation, elt_clip_min,                                \
            fuse_param.has_elt_clip,       elt_clip_max,                                \
            fuse_param.has_concat,         concat_offset_v8,                            \
            concat_stride_v8

#define MERGE_KPARAM_LIST \
        	conv_out,                      final_out,                                   \
        	spk_height_v1,                 spk_width_v8,                                \
        	out_hw,                        splitk * splitf,                             \
            conv_param.has_bias,           bias,                                        \
            fuse_param.has_activation,     clip_min,                                    \
            fuse_param.has_clip,           clip_max,                                    \
            fuse_param.has_elt,            (const int4 *) fuse_param.pre_data,          \
            fuse_param.has_elt_activation, elt_clip_min,                                \
            fuse_param.has_elt_clip,       elt_clip_max,                                \
            fuse_param.has_concat,         concat_offset_v8,                            \
            concat_stride_v8

static std::vector<kernel_info_t> g_kernel_container;
static bool is_g_kernel_container_initialized = false;

static std::unordered_map<size_t, algo_param_t> g_conv_shape_hash;

void InitializeKernelContainer(std::vector<kernel_info_t> &g_kernel_container, ppl::common::datatype_t type)
{
    if( type == ppl::common::DATATYPE_FLOAT16 ) {
        Initialize2spkConvF1KernelContainer(g_kernel_container);
        Initialize2spkConvF3KernelContainer(g_kernel_container);
        Initialize2spkConvFNKernelContainer(g_kernel_container);
        Initialize2spkConvFSKernelContainer(g_kernel_container);
                      
        InitializeIdxnConvKernelContainer(g_kernel_container);

        InitializeSwzlConvF1KernelContainer(g_kernel_container);
        InitializeSwzlConvF3KernelContainer(g_kernel_container);
        InitializeSwzlConvFNKernelContainer(g_kernel_container);
    }
    
    is_g_kernel_container_initialized = true;
}

__inline__ std::string GetConvShapeString(conv_param_t &conv_param)
{
    return std::string("b" + std::to_string(conv_param.in_num)  + \
                       "_c" + std::to_string(conv_param.num_chl) + \
                       "_d" + std::to_string(conv_param.num_flt) + \
                       "_g" + std::to_string(conv_param.num_grp) + \
                       "_h" + std::to_string(conv_param.in_height) + \
                       "_w" + std::to_string(conv_param.in_width) + \
                       "_r" + std::to_string(conv_param.flt_height) + \
                       "_s" + std::to_string(conv_param.flt_width) + \
                       "_p" + std::to_string(conv_param.pad_height) + \
                       "_q" + std::to_string(conv_param.pad_width) + \
                       "_u" + std::to_string(conv_param.stride_height) + \
                       "_v" + std::to_string(conv_param.stride_width) + \
                       "_y" + std::to_string(conv_param.hole_height) + \
                       "_x" + std::to_string(conv_param.hole_width) + \
                       "_");
}

__inline__ size_t GetConvShapeHashKey( conv_param_t &conv_param )
{
    return std::hash<std::string>{} (GetConvShapeString(conv_param));
}

uint64_t PPLCUDAConvolutionGetCompilationBufSize(ppl::common::datatype_t type, conv_param_t &conv_param, uint64_t workspace)
{
    int pad_size = GetPadSize(type);

    uint32_t num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    uint32_t num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    uint32_t num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    uint32_t num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size); 

    bool  is_in_grp_pad = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;

    uint32_t cvt_input_size = 0;
    uint32_t cvt_output_size = 0;

    if(is_in_grp_pad)
        cvt_input_size = GetCvtInputSize( type, conv_param, num_chl_per_grp_pad);

    if(is_out_grp_pad)
        cvt_output_size = getCvtOutputSize(type, conv_param, num_flt_per_grp_pad);

    uint32_t split_size = GetMaxSplitSize(type, conv_param, num_flt_per_grp_pad);

    uint64_t total_size = cvt_input_size + cvt_output_size + split_size;

    return total_size <= workspace ? total_size : workspace;
}
uint64_t PPLCUDAConvolutionGetRuntimeBufSize(
        ppl::common::datatype_t type,
        conv_param_t &conv_param,
        unsigned int splitk,
        unsigned int splitf,
        uint64_t workspace)
{
    int pad_size = GetPadSize(type);

    uint32_t num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    uint32_t num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    uint32_t num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    uint32_t num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size); 

    bool  is_in_grp_pad = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;

    uint32_t cvt_input_size = 0;
    uint32_t cvt_output_size = 0;

    if(is_in_grp_pad)
        cvt_input_size = GetCvtInputSize(type, conv_param, num_chl_per_grp_pad);
    if(is_out_grp_pad)
        cvt_output_size = getCvtOutputSize(type, conv_param, num_flt_per_grp_pad);

    uint32_t split_size = 0;
    
    if(splitk > 1 || splitf > 1)
        split_size = GetSplitKFSize(type, conv_param, num_flt_per_grp_pad, splitk, splitf);

    uint64_t total_size  = cvt_input_size + cvt_output_size + split_size;

    return total_size <= workspace ? total_size : workspace;
}

ppl::common::RetCode PPLCUDAConvolutionQuickSelectKernel(
        select_param_t &select_param,
        conv_param_t &conv_param) {
    int in_hw = conv_param.in_num * conv_param.in_height * conv_param.in_width;
    int out_hw = conv_param.in_num * conv_param.out_height * conv_param.out_width;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    int chl_per_group = conv_param.num_chl / conv_param.num_grp;

    // Use non-shared memory algo for small channel
    if (chl_per_group < 64) {
        if (flt_hw > 9) {
            select_param.m_cta = 128;
            select_param.m_warp = 64;
        } else {
            select_param.m_cta = 32;
            select_param.m_warp = 16;
        }

        if (in_hw == out_hw) {
            select_param.n_cta = 64;
            select_param.n_warp = 32;
        } else {
            select_param.n_cta = 32;
            select_param.n_warp = 16;
        }

        if (conv_param.num_chl >= 16) {
            select_param.k_cta = 32;
            select_param.k_warp = 32;
        } else {
            select_param.k_cta = 16;
            select_param.k_warp = 16;
        }
    } else { // Use 3spk algo for large channel
        float min_pad = 1.0;
        select_param.m_cta = 16;
        for (int32_t i = 128; i >= 16; i = i / 2) {
            if (out_hw < i) continue;
            float pad = 1.0 * (DivUp(out_hw, i) * i - out_hw) / out_hw;
            if (pad < min_pad)  {
                min_pad = pad;
                select_param.m_cta = i;
            }
            if (min_pad < 0.1)  break;
        }

        select_param.n_cta = 16;
        for (int32_t i = 128; i >= 16; i = i / 2) {
            int cout = conv_param.num_flt;
            if ((cout < 64 && i / cout == 1) || (cout >= 64 && cout % i == 0)) {
                select_param.n_cta = i;
                break;
            }
        }

        if (conv_param.num_chl >= 128) {
            select_param.k_cta = 64;
        } else {
            select_param.k_cta = 32;
        }

        if (select_param.m_cta == 128 && select_param.n_cta == 128) {
            select_param.m_cta = 64;
        }

        if (select_param.m_cta * 4 < select_param.n_cta) {
            select_param.m_cta *= 2;
            select_param.n_cta /= 2;
        }
        if (select_param.n_cta *4 < select_param.m_cta) {
            select_param.m_cta /= 2;
            select_param.n_cta *= 2;
        }

        select_param.m_warp = select_param.m_cta / 2;
        select_param.n_warp = select_param.n_cta / 2;
        select_param.k_warp = select_param.k_cta / 2;
        if (select_param.k_warp < 8) {
            select_param.k_warp = 16;
        }
    }
    select_param.quick_select = true;
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAConvolutionSelectKernel(
        cudaStream_t &stream, 
        ppl::common::datatype_t type,
        int4* d_input,
        int4* d_flt,
        int4* d_output,
        int4* bias,
        int4* d_temp_buf, 
        algo_param_t & algo_param,
        conv_param_t &conv_param, 
        fuse_param_t &fuse_param,
        select_param_t &select_param,
        std::string node_name,
	    uint64_t workspace)
{

    int selected_count = 0;
    // ofstream outfile;
    // outfile.open(node_name + ".csv");

    if(!is_g_kernel_container_initialized)
        InitializeKernelContainer(g_kernel_container, type);

    size_t conv_shape_hash = GetConvShapeHashKey(conv_param);

    std::unordered_map<size_t, algo_param_t>::const_iterator conv_shape_hash_iterator = g_conv_shape_hash.find(conv_shape_hash);

    if(conv_shape_hash_iterator != g_conv_shape_hash.end()) {
        algo_param.kid    = conv_shape_hash_iterator->second.kid;
        algo_param.splitk = conv_shape_hash_iterator->second.splitk;
        algo_param.splitf = conv_shape_hash_iterator->second.splitf;

        return ppl::common::RC_SUCCESS;
    }

    int pad_size = GetPadSize(type);

    int num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    int num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    int num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    int in_hw = conv_param.in_height * conv_param.in_width;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    int out_hw = conv_param.out_height * conv_param.out_width;

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;

    bool  is_in_grp_pad = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;

    uint64_t buf_off_v4 = 0;

    int4 *pad_input = d_input;
    int4 *pad_output = d_output;

    if(is_in_grp_pad) {
	    pad_input = d_temp_buf; 
	    buf_off_v4 += GetCvtInputSize(type, conv_param, num_chl_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);

        PPLCUDAConvolutionCvtInput(stream, pad_input, d_input, type, conv_param);
    }

    if(is_out_grp_pad) {
	    pad_output = d_temp_buf + buf_off_v4;
	    buf_off_v4 += getCvtOutputSize(type, conv_param, num_flt_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);
    } 

    int4 * final_out = fuse_param.has_concat ? (int4 *) fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;

    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    float minTime = FLT_MAX;

    float elapsed;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    const int SPLITK_OPTIONS[] = {1, 2, 4, 8};
    for(unsigned int spk = 0; spk < 4; spk++) {
        unsigned int splitk = SPLITK_OPTIONS[spk];

        for(unsigned int kid = 0; kid < g_kernel_container.size(); kid++) {
            unsigned int splitf = (g_kernel_container[kid].ktype == CONV_2SPK_FS) ? flt_hw : 1;
        
            if(!g_kernel_container[kid].CheckKernelTypeFeasible(conv_param.flt_height, conv_param.flt_width, num_chl_per_grp, splitk)) continue;

            if(!g_kernel_container[kid].CheckSplitkFeasible(num_chl_per_grp, splitk)) continue;

            if(!g_kernel_container[kid].CheckSplitfFeasible(splitf, splitk)) continue;

            if(!g_kernel_container[kid].CheckQuickSelectFeasible(select_param, conv_param.num_chl / conv_param.num_grp, splitk, splitf)) continue;

            int4 *conv_out = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

            dim3 block_size, grid_size;

            block_size.x = g_kernel_container[kid].cta_size_in_thd;
            block_size.y = 1;
            block_size.z = 1;

            if(g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || \
                    g_kernel_container[kid].ktype == CONV_SWZL_FN) {
                grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_kernel_container[kid].tile_n_per_cta);
                grid_size.y = DivUp(num_flt_per_grp_pad, g_kernel_container[kid].tile_m_per_cta);
            } else {
                grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_kernel_container[kid].tile_m_per_cta);
                grid_size.y = DivUp(num_flt_per_grp_pad, g_kernel_container[kid].tile_n_per_cta);
            }

            grid_size.z = conv_param.num_grp * splitk * splitf;

            if(!g_kernel_container[kid].CheckSplitkBlockSizeFeasible((grid_size.x * grid_size.y * grid_size.z), splitk)) continue;

	        cudaEventRecord(begin, stream);

	        for(int i = 0; i < TIMES; i++) {
                if(g_kernel_container[kid].ktype == CONV_IDXN_C2 || g_kernel_container[kid].ktype == CONV_IDXN_C4 || \
                        g_kernel_container[kid].ktype == CONV_IDXN_C32) {
                    int tile_k_per_step = g_kernel_container[kid].tile_k_per_step;

                    int img_pad_size    = pad_size;
                    int flt_pad_size    = g_kernel_container[kid].flt_pad_size;
                    int out_nhw         = out_hw * conv_param.in_num;

                    int in_chl_per_grp_pad = Align(num_chl_per_grp, img_pad_size);
                    int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
                    int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

	                int kloop_num        = DivUp(flt_hw * flt_chl_per_grp_pad, g_kernel_container[kid].tile_k_per_cta);
                    int koff_num_pad      = Align(kloop_num * (g_kernel_container[kid].tile_k_per_cta / flt_pad_size), WARP_SIZE);

                    (g_kernel_container[kid].idx_kptr)<<<grid_size, block_size, 0, stream>>>(IDX_KPARAM_LIST);
                }
                else if(g_kernel_container[kid].ktype == CONV_2SPK_F1 || g_kernel_container[kid].ktype == CONV_2SPK_F3 || \
                        g_kernel_container[kid].ktype == CONV_2SPK_FN || g_kernel_container[kid].ktype == CONV_2SPK_FS || \
                        g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || \
                        g_kernel_container[kid].ktype == CONV_SWZL_FN) {

	                int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, g_kernel_container[kid].tile_k_per_cta);

                    lut_t in_lut, flt_lut;
                    int in_lut_size, flt_lut_size;
                
                    InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height,
                            conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width,
                            num_chl_per_grp_pad, conv_param.num_grp, g_kernel_container[kid].tile_k_per_cta, pad_size);

                    InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad,
                            g_kernel_container[kid].tile_k_per_cta, pad_size);

                    if(splitk == 1) {
                        if(g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || \
                                g_kernel_container[kid].ktype == CONV_SWZL_FN)
                            (g_kernel_container[kid].lut_kptr)<<<grid_size, block_size, 0, stream>>>(SWZL_LUT_KPARAM_LIST);
                        else
                            (g_kernel_container[kid].lut_kptr)<<<grid_size, block_size, 0, stream>>>(LUT_KPARAM_LIST);
                    } else {
                        int num_chl_per_spk_head, num_chl_per_spk_tail;

                        InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, 
                                pad_size, g_kernel_container[kid].tile_k_per_cta, splitk);

                        if(g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || \
                                g_kernel_container[kid].ktype == CONV_SWZL_FN)
                            (g_kernel_container[kid].spk_kptr)<<<grid_size, block_size, 0, stream>>>(SWZL_SPK_KPARAM_LIST);
                        else
                            (g_kernel_container[kid].spk_kptr)<<<grid_size, block_size, 0, stream>>>(SPK_KPARAM_LIST);
                    }

                    if(splitk > 1 || splitf > 1) {
                        int spk_width_v8   = num_flt_per_grp_pad * conv_param.num_grp / pad_size;
                        int spk_height_v1  = out_hw * conv_param.in_num;

                        dim3 merge_grid_size, merge_block_size;
                        merge_block_size.x = 64; // empirical value
                        merge_block_size.y = 1;
                        merge_block_size.z = 1;

                        merge_grid_size.x  = spk_height_v1;
                        merge_grid_size.y  = DivUp(spk_width_v8, merge_block_size.x);
                        merge_grid_size.z  = 1;

                        MergeConvSplitResults<<<merge_grid_size, merge_block_size, 0, stream>>>(MERGE_KPARAM_LIST);
                    }
                }
            }

	        cudaEventRecord(end, stream);
	        cudaEventSynchronize(end);
	        cudaEventElapsedTime(&elapsed, begin, end);

	        if(elapsed < minTime){
                algo_param.kid = kid;
                algo_param.splitk = splitk;
                algo_param.splitf = splitf;
	            minTime = elapsed;
	        }

            // outfile << kid << "," << elapsed << ","
            //     << in_hw << ","
            //     << flt_hw << ","
            //     << out_hw << ","
            //     << conv_param.num_flt << ","
            //     << conv_param.num_chl << ","
            //     << g_kernel_container[kid].tile_m_per_cta << ","
            //     << g_kernel_container[kid].tile_n_per_cta << ","
            //     << g_kernel_container[kid].tile_k_per_cta << ","
            //     << g_kernel_container[kid].tile_m_per_warp << ","
            //     << g_kernel_container[kid].tile_n_per_warp << ","
            //     << g_kernel_container[kid].tile_k_per_warp << ","
            //     << g_kernel_container[kid].tile_k_per_step << ","
            //     << g_kernel_container[kid].tile_k_per_set << ","
            //     << g_kernel_container[kid].flt_size << ","
            //     << g_kernel_container[kid].flt_pad_size << ","
            //     << g_kernel_container[kid].cta_size_in_thd << ","
            //     << g_kernel_container[kid].kname << ","
            //     << splitk << "," << splitf << endl;
            if (splitk == 1)   selected_count++;
        }
    }

    if(is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);

    g_conv_shape_hash[conv_shape_hash] = algo_param;
    printf("%s,%d,%d,%d\n", GetConvShapeString(conv_param).data(), algo_param.kid, algo_param.splitk, algo_param.splitf);
    // outfile.close();

    return ppl::common::RC_SUCCESS;
}

void PPLCUDAConvolutionForwardImp(
        cudaStream_t &stream, 
        ppl::common::datatype_t type,
        int4* d_input,
        int4* d_flt,
        int4* d_output,
        int4* bias,
        int4* d_temp_buf,
        algo_param_t& algo_param,
        conv_param_t &conv_param,
        fuse_param_t &fuse_param)
{
    if(!is_g_kernel_container_initialized)
        InitializeKernelContainer(g_kernel_container, type);

    unsigned int kid = algo_param.kid;
    unsigned int splitk = algo_param.splitk;
    unsigned int splitf = algo_param.splitf;

    int pad_size = GetPadSize(type);

    int num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    int num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    int num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    int in_hw  = conv_param.in_height * conv_param.in_width;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    int out_hw = conv_param.out_height * conv_param.out_width;

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;

    bool  is_in_grp_pad = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;

    uint64_t buf_off_v4 = 0;

    int4 *pad_input = d_input;
    int4 *pad_output = d_output;

    if(is_in_grp_pad) {
	    pad_input = d_temp_buf; 
	    buf_off_v4 += GetCvtInputSize(type, conv_param, num_chl_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);

        PPLCUDAConvolutionCvtInput(stream, pad_input, d_input, type, conv_param);
    }

    if(is_out_grp_pad) {
	    pad_output = d_temp_buf + buf_off_v4;
	    buf_off_v4 += getCvtOutputSize(type, conv_param, num_flt_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);
    } 

    int4 *final_out  = fuse_param.has_concat ? (int4 *) fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;
    int4 *conv_out   = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    dim3 block_size, grid_size;

    block_size.x = g_kernel_container[kid].cta_size_in_thd;
    block_size.y = 1;
    block_size.z = 1;

    if(g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || \
            g_kernel_container[kid].ktype == CONV_SWZL_FN) {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_kernel_container[kid].tile_n_per_cta);
        grid_size.y = DivUp(num_flt_per_grp_pad, g_kernel_container[kid].tile_m_per_cta);
    } else {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_kernel_container[kid].tile_m_per_cta);
        grid_size.y = DivUp(num_flt_per_grp_pad, g_kernel_container[kid].tile_n_per_cta);
    }

    grid_size.z = conv_param.num_grp * splitk * splitf;

    if(g_kernel_container[kid].ktype == CONV_IDXN_C2 || g_kernel_container[kid].ktype == CONV_IDXN_C4 || \
            g_kernel_container[kid].ktype == CONV_IDXN_C32) {
        int img_pad_size = pad_size;
        int flt_pad_size = g_kernel_container[kid].flt_pad_size;

        int out_nhw = out_hw * conv_param.in_num;

        int in_chl_per_grp_pad = Align(num_chl_per_grp, img_pad_size);
        int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
        int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

	    int kloop_num = DivUp(flt_hw * flt_chl_per_grp_pad, g_kernel_container[kid].tile_k_per_cta);
        int koff_num_pad = Align(kloop_num * (g_kernel_container[kid].tile_k_per_cta / flt_pad_size), WARP_SIZE);

        (g_kernel_container[kid].idx_kptr)<<<grid_size, block_size, 0, stream>>>(IDX_KPARAM_LIST);

    } else if(g_kernel_container[kid].ktype == CONV_2SPK_F1 || g_kernel_container[kid].ktype == CONV_2SPK_F3 || \
              g_kernel_container[kid].ktype == CONV_2SPK_FN || g_kernel_container[kid].ktype == CONV_2SPK_FS || \
              g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || \
              g_kernel_container[kid].ktype == CONV_SWZL_FN) {

	    int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, g_kernel_container[kid].tile_k_per_cta);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;
    
        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height,
                conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width,
                num_chl_per_grp_pad, conv_param.num_grp, g_kernel_container[kid].tile_k_per_cta, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad,
                g_kernel_container[kid].tile_k_per_cta, pad_size);

        if(splitk == 1) {
            if(g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || \
                    g_kernel_container[kid].ktype == CONV_SWZL_FN)
                (g_kernel_container[kid].lut_kptr)<<<grid_size, block_size, 0, stream>>>(SWZL_LUT_KPARAM_LIST);
            else
                (g_kernel_container[kid].lut_kptr)<<<grid_size, block_size, 0, stream>>>(LUT_KPARAM_LIST);
        } else {
            int num_chl_per_spk_head, num_chl_per_spk_tail;

            InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, 
                    pad_size, g_kernel_container[kid].tile_k_per_cta, splitk);

            if(g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || \
                    g_kernel_container[kid].ktype == CONV_SWZL_FN)
                (g_kernel_container[kid].spk_kptr)<<<grid_size, block_size, 0, stream>>>(SWZL_SPK_KPARAM_LIST);
            else
                (g_kernel_container[kid].spk_kptr)<<<grid_size, block_size, 0, stream>>>(SPK_KPARAM_LIST);
        }
    }
    
    if(splitk > 1 || splitf > 1) {
        int spk_width_v8   = num_flt_per_grp_pad * conv_param.num_grp / pad_size;
        int spk_height_v1  = out_hw * conv_param.in_num;

        dim3 merge_grid_size, merge_block_size;
        merge_block_size.x = 64;
        merge_block_size.y = 1;
        merge_block_size.z = 1;

        merge_grid_size.x  = spk_height_v1;
        merge_grid_size.y  = DivUp(spk_width_v8, merge_block_size.x);
        merge_grid_size.z  = 1;

        MergeConvSplitResults<<<merge_grid_size, merge_block_size, 0, stream>>>(MERGE_KPARAM_LIST);
    }

    if(is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }
    
}
