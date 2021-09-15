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

//////////////////////////////////////////////////////
// half output interface
//////////////////////////////////////////////////////

#if defined(ENABLE_FUSE)

#define OUTPUT_BY_INT1() \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv1_y_valid[0] && dCv1_x_valid[i] ) dCv1[concat_v1_off0 + dCv1_idx[i]] = C[Cv1_off + i]; \
                \
                if( dCv1_y_valid[1] && dCv1_x_valid[i] ) dCv1[concat_v1_off1 + dCv1_idx[i]] = C[Cv1_off + i + NUM_N_STEPS]; \
            } \
            \
            dCv1_y_valid[0] = (dCv1_idy[0] < out_nhw); \
            dCv1_y_valid[1] = (dCv1_idy[1] < out_nhw); \
            dCv1_idy[0]  += TILE_M_PER_STEP; \
            dCv1_idy[1]  += TILE_M_PER_STEP; \
        }

#else

#define OUTPUT_BY_INT1() \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv1_y_valid[0] && dCv1_x_valid[i] ) dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[i]] = C[Cv1_off + i]; \
                \
                if( dCv1_y_valid[1] && dCv1_x_valid[i] ) dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[i]] = C[Cv1_off + i + NUM_N_STEPS]; \
            } \
            \
            dCv1_y_valid[0] = (dCv1_idy[0] < out_nhw); \
            dCv1_y_valid[1] = (dCv1_idy[1] < out_nhw); \
            dCv1_idy[0]  += TILE_M_PER_STEP; \
            dCv1_idy[1]  += TILE_M_PER_STEP; \
        }

#endif

//////////////////////////////////////////////////////
// bias macros
//////////////////////////////////////////////////////

#define LOAD_BIAS_V1(_has_bias, _bias, _r_bias_v1) \
        { \
             _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
                if( _has_bias && dCv1_x_valid[i] ) _r_bias_v1[i] = ((int *) _bias) [dCv1_idx[i]]; \
        }

#define ADD_BIAS_V1(_has_bias, _r_bias_v1) \
        { \
            if( _has_bias ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] ) HADD2_INST(C[Cv1_off + i],               C[Cv1_off + i],               _r_bias_v1[i]); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] ) HADD2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], _r_bias_v1[i]); \
                } \
            } \
        }

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define FUSE_CLIP_V1(_has_clip, _clip_max, _clip_min) \
        { \
	        if( _has_clip ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] ) HMIN2_INST(C[Cv1_off + i],               C[Cv1_off + i],               _clip_max, C[Cv1_off + i]); \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i],               C[Cv1_off + i],               _clip_min, C[Cv1_off + i]); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] ) HMIN2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], _clip_max, C[Cv1_off + i + NUM_N_STEPS]); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], _clip_min, C[Cv1_off + i + NUM_N_STEPS]); \
	            } \
	        } \
        }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define FUSE_RELU_V1(_has_relu) \
        { \
	        if( _has_relu ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i],               C[Cv1_off + i],               0, C[Cv1_off + i]); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], 0, C[Cv1_off + i + NUM_N_STEPS]); \
	            } \
	        } \
        }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define LOAD_ELT_V1(_has_elt, _pre_data, _r_elt_v1) \
        { \
	        if( _has_elt ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if(dCv1_y_valid[0] && dCv1_x_valid[i] ) _r_elt_v1[i]               = ((int *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[i]]; \
                    if(dCv1_y_valid[1] && dCv1_x_valid[i] ) _r_elt_v1[i + NUM_N_STEPS] = ((int *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[i]];  \
                } \
            } \
        }

#define FUSE_ELT_V1(_has_elt, _r_elt_v1) \
        { \
	        if( _has_elt ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                     HADD2_INST(C[Cv1_off + i],               C[Cv1_off + i],               _r_elt_v1[i]); \
                     HADD2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], _r_elt_v1[i + NUM_N_STEPS]); \
	            } \
	        } \
        }

//////////////////////////////////////////////////////
// concat macros
//////////////////////////////////////////////////////

#define SET_CONCAT_OFF_V1(_has_concat, _concat_v1_off0, _concat_v1_off1) \
        { \
            _concat_v1_off0 = (_has_concat) ? (concat_offset_v8 + dCv1_idy[0] * concat_stride_v8) * _INT4_TO_4INT_ : dCv1_idy[0] * num_flt_v2; \
            _concat_v1_off1 = (_has_concat) ? (concat_offset_v8 + dCv1_idy[1] * concat_stride_v8) * _INT4_TO_4INT_ : dCv1_idy[1] * num_flt_v2; \
        }
