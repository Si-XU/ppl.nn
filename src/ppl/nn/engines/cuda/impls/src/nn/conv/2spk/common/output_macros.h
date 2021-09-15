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

#if defined(ENABLE_FUSE)

#define OUTPUT_PRC_HALF(_Rv4) \
        { \
            if( dCv4_x_valid && dCv4_y_valid ) \
                dC[dCv4_base + concat_v4_off] = _Rv4[0]; \
            \
            dCv4_idy   +=  OUTPUT_SIZE_Y_IN_THD; \
            dCv4_y_valid  = (dCv4_idy / out_hw) < in_num; \
        }

#else

#define OUTPUT_PRC_HALF(_Rv4) \
        { \
            if( dCv4_x_valid && dCv4_y_valid ) \
                dC[dCv4_base + dCv4_idy * num_flt_per_grp_pad_v8 * num_grp] = _Rv4[0]; \
            \
            dCv4_idy   +=  OUTPUT_SIZE_Y_IN_THD; \
            dCv4_y_valid  = (dCv4_idy / out_hw) < in_num; \
        }
#endif

//////////////////////////////////////////////////////
// bias macros
//////////////////////////////////////////////////////

#define LOAD_BIAS_V4(_has_bias, _bias, _bias_v4_off) \
        { \
            if( _has_bias && dCv4_x_valid ) \
                Rv4[_bias_v4_off + 0] = ((int4 *) _bias) [grp_id * num_flt_per_grp_pad_v8 + dCv4_idx]; \
        }

#define ADD_BIAS_V4(_has_bias, _bias_v1_off) \
        { \
            if(_has_bias) \
            { \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                { \
                    HADD2_INST(R[i], R[_bias_v1_off + i], R[i]); \
	            } \
            } \
        }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define FUSE_RELU_V4(_has_relu) \
        { \
	        if(_has_relu) \
            { \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                { \
                    HMAX2_INST(R[i], R[i], 0, R[i]); \
	            } \
		    } \
        }

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define FUSE_CLIP_V4(_has_clip, _clip_max, _clip_min) \
        { \
	        if(_has_clip) \
            { \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                { \
                    HMIN2_INST(R[i], R[i], _clip_max, R[i]); \
                    HMAX2_INST(R[i], R[i], _clip_min, R[i]); \
	            } \
		    } \
        }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define LOAD_ELT_V4(_has_elt, _pre_data, _elt_v4_off) \
        { \
       	    if( _has_elt && dCv4_y_valid && dCv4_x_valid ) \
                Rv4[_elt_v4_off] = ((int4 *) _pre_data) [dCv4_base + dCv4_idy * num_flt_per_grp_pad_v8 * num_grp]; \
        }

#define FUSE_ELT_V4(_has_elt, _elt_v1_off) \
        { \
	        if( _has_elt ) \
            { \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++){ \
                    HADD2_INST(R[i], R[_elt_v1_off + i], R[i]); \
	            } \
	        } \
        }

#define SET_CONCAT_OFF_V4(_has_concat, _concat_v4_off) \
        { \
	        _concat_v4_off = (_has_concat) ? dCv4_idy * concat_stride_v8 + concat_offset_v8 : dCv4_idy * num_flt_per_grp_pad_v8 * num_grp; \
        }
        
