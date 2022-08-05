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

#ifndef __PPLCUDA_CONV_KERNEL_TYPE_H__
#define __PPLCUDA_CONV_KERNEL_TYPE_H__

#include <cuda.h>
#include <cuda_fp16.h>

#include "common/init_lut.h"

#define MAX_VEC_SIZE 6

struct int_vec_t {
    int idx[MAX_VEC_SIZE];
    int_vec_t(){};
};

struct bool_vec_t {
    bool idx[MAX_VEC_SIZE];
    bool_vec_t(){};
};

struct half_vec_t {
    __half idx[MAX_VEC_SIZE];
    half_vec_t(){};
};

struct half2_vec_t {
    __half2 idx[MAX_VEC_SIZE];
    half2_vec_t(){};
};

struct int4_addr_vec_t {
    int4 * idx[MAX_VEC_SIZE];
    int4_addr_vec_t(){};
};

struct int4_addr_vec_t {
    int4 * idx[MAX_VEC_SIZE];
    int4_addr_vec_t(){};
};

struct void_addr_vec_t {
    void * idx[MAX_VEC_SIZE];
    void_addr_vec_t(){};
};

typedef void lut_kernel_t(
    int4_addr_vec_t dA_vec,
    int4_addr_vec_t dB_vec,
    int4_addr_vec_t dC_vec,
    int hconv_num,
    int_vec_t hconv_vec,
    int kloop_num,
    struct lut_t in_lut,  int in_lut_size,
    struct lut_t flt_lut, int flt_lut_size,
    int in_hw,  int out_hw,
    int flt_hw, int splitk,
    int in_height, int in_width,
    int in_num,    int num_grp,
    int num_chl_per_grp, int num_chl_per_grp_pad,
    int flt_height, int flt_width,
    int_vec_t num_flt_per_grp_vec,
    int_vec_t num_flt_per_grp_pad_vec,
    int out_height,    int out_width,
    int stride_height, int stride_width,
    int pad_height,    int pad_width,
    int hole_height,   int hole_width,
    int_vec_t has_bias_vec,
    int4_addr_vec_t bias_vec,
    int_vec_t has_relu_vec,
    bool_vec_t has_clip_vec,
    half2_vec_t clip_min_vec,
    half2_vec_t clip_max_vec,
    int_vec_t has_prelu_vec,
    void_addr_vec_t prelu_vec,
    bool_vec_t has_elt_vec,
    int4_addr_vec_t pre_data_vec,
    int_vec_t has_elt_relu_vec,
    bool_vec_t has_elt_clip_vec,
    half2_vec_t elt_clip_min_vec,
    half2_vec_t elt_clip_max_vec,
    int_vec_t has_elt_prelu_vec,
    void_addr_vec_t elt_prelu_vec,
    half_vec_t leaky_vec,
    half_vec_t elt_leaky_vec,
    bool_vec_t has_concat_vec,
    int_vec_t concat_offset_v8_vec,
    int_vec_t concat_stride_v8_vec);

typedef void spk_kernel_t(
    int4_addr_vec_t dA_vec,
    int4_addr_vec_t dB_vec,
    int4_addr_vec_t dC_vec,
    int hconv_num,
    int_vec_t hconv_vec,
    int kloop_num,
    struct lut_t in_lut,  int in_lut_size,
    struct lut_t flt_lut, int flt_lut_size,
    int num_chl_per_spk_head,
    int num_chl_per_spk_tail,
    int in_hw,  int out_hw,
    int flt_hw, int splitk,
    int in_height, int in_width,
    int in_num,    int num_grp,
    int num_chl_per_grp, int num_chl_per_grp_pad,
    int flt_height, int flt_width,
    int_vec_t num_flt_per_grp_vec,
    int_vec_t num_flt_per_grp_pad_vec,
    int out_height,    int out_width,
    int stride_height, int stride_width,
    int pad_height,    int pad_width,
    int hole_height,   int hole_width,
    int_vec_t has_bias_vec,
    int4_addr_vec_t bias_vec);

typedef void idx_kernel_t(
    int4_addr_vec_t dA_vec,
    int4_addr_vec_t dB_vec,
    int4_addr_vec_t dC_vec,
    int hconv_num,
    int_vec_t hconv_vec,
    int kloop_num, int koff_num_pad,
    int in_hw,     int out_hw,
    int flt_hw,    int out_nhw,
    int in_height, int in_width,
    int in_num,    int num_grp,
    int num_chl,   int num_chl_per_grp,
    int in_chl_per_grp_pad, int flt_chl_per_grp_pad,
    int flt_height, int flt_width,
    int_vec_t num_flt_per_grp_vec,
    int_vec_t num_flt_per_grp_pad_vec,
    int out_height,    int out_width,
    int stride_height, int stride_width,
    int pad_height,    int pad_width,
    int hole_height,   int hole_width,
    int_vec_t has_bias_vec,
    int4_addr_vec_t bias_vec,
    int_vec_t has_relu_vec,
    bool_vec_t has_clip_vec,
    half2_vec_t clip_min_vec,
    half2_vec_t clip_max_vec,
    int_vec_t has_prelu_vec,
    void_addr_vec_t prelu_vec,
    bool_vec_t has_elt_vec,
    int4_addr_vec_t pre_data_vec,
    int_vec_t has_elt_relu_vec,
    bool_vec_t has_elt_clip_vec,
    half2_vec_t elt_clip_min_vec,
    half2_vec_t elt_clip_max_vec,
    int_vec_t has_elt_prelu_vec,
    void_addr_vec_t elt_prelu_vec,
    half_vec_t leaky_vec,
    half_vec_t elt_leaky_vec,
    bool_vec_t has_concat_vec,
    int_vec_t concat_offset_v8_vec,
    int_vec_t concat_stride_v8_vec);

typedef void int8_lut_kernel_t(
        int4* dA,               
        int4* dB,
        int4* dC,
        int kloop_num,
        struct lut_t in_lut,          int in_lut_size,
        struct lut_t flt_lut,         int flt_lut_size,
        int  in_hw,                   int out_hw,
        int  flt_hw,                  int splitk,
        int  in_height,               int in_width,
        int  in_num,                  int num_grp,
        int  num_chl_per_grp,         int num_chl_per_grp_pad,
        int  flt_height,              int flt_width,
        int  num_flt_per_grp,         int num_flt_per_grp_pad,
        int  out_height,              int out_width,
        int  stride_height,           int stride_width,
        int  pad_height,              int pad_width,
        int  hole_height,             int hole_width,
        int  has_bias,                const int4* bias,
	float in_scale,               void *d_flt_scale,
        float out_scale,              float pre_scale,
        int  has_relu,                const float clip_min,
	bool has_clip,                const float clip_max,
        int  has_prelu,               const void* prelu,
        bool has_elt,                 const int4* pre_data,
        int  has_elt_relu,            const float elt_clip_min,
	bool has_elt_clip,            const float elt_clip_max,
        int has_elt_prelu,            const void* elt_prelu,
        const float leaky,           const float elt_leaky,
        bool has_concat,              int concat_offset_v16,
        int concat_stride_v16);

typedef void int8_spk_kernel_t(
        int4* dA,
        int4* dB,
        int4* dC,
        int kloop_num,
        struct lut_t in_lut,          int in_lut_size,
        struct lut_t flt_lut,         int flt_lut_size,
        int num_chl_per_spk_head,
        int num_chl_per_spk_tail,
        int in_hw,                    int out_hw,
        int flt_hw,                   int splitk,
        int in_height,                int in_width,
        int in_num,                   int num_grp,
        int num_chl_per_grp,          int num_chl_per_grp_pad,
        int flt_height,               int flt_width,
        int num_flt_per_grp,          int num_flt_per_grp_pad,
        int out_height,               int out_width,
        int stride_height,            int stride_width,
        int pad_height,               int pad_width,
        int hole_height,              int hole_width,
        int has_bias,                 int* bias,
	    float in_scale,               void *d_flt_scale);

typedef void int8_idx_kernel_t(
        int4* dA,
        int4* dB,
        int4* dC,
        int  kloop_num,               int koff_num_pad,
        int  in_hw,                   int out_hw,
        int  flt_hw,                  int out_nhw,
        int  in_height,               int in_width,
        int  in_num,                  int num_grp,
        int  num_chl,                 int num_chl_per_grp,
        int  in_chl_per_grp_pad,      int flt_chl_per_grp_pad,
        int  flt_height,              int flt_width,
        int  num_flt_per_grp,         int num_flt_per_grp_pad,
        int  out_height,              int out_width,
        int  stride_height,           int stride_width,
        int  pad_height,              int pad_width,
        int  hole_height,             int hole_width,
        int  has_bias,                const int4* bias,
	float in_scale,               void *d_flt_scale,
        float out_scale,              float pre_scale,
        int  has_relu,                const float clip_min,
	bool has_clip,                const float clip_max,
        int  has_prelu,               const void* prelu,
        bool has_elt,                 const int4* pre_data,
        int  has_elt_relu,            const float elt_clip_min,
	bool has_elt_clip,            const float elt_clip_max,
        int  has_elt_prelu,           const void* elt_prelu,
        const float leaky,            const float elt_leaky,
        bool has_concat,              int concat_offset_v16,
        int concat_stride_v16);

#endif
