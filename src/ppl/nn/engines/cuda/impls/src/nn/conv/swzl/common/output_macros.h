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

#define OUTPUT_BY_INT4(_Rv4) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
            { \
                if( dCv4_y_valid && dCv4_x_valid[i] ) \
                    dC[dCv4_base + concat_v4_off[i]] = _Rv4[i]; \
                \
                dCv4_idx[i]   +=  OUTPUT_SIZE_X_IN_THD * OUTPUT_BLKS_PER_STEP; \
                dCv4_x_valid[i]  = (dCv4_idx[i] / out_hw) < in_num; \
            } \
        }

#else

#define OUTPUT_BY_INT4(_Rv4) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
            { \
                if( dCv4_y_valid && dCv4_x_valid[i] ) \
                    dC[dCv4_base + dCv4_idx[i] * num_flt_per_grp_pad_v8 * num_grp] = _Rv4[i]; \
                \
                dCv4_idx[i]   +=  OUTPUT_SIZE_X_IN_THD * OUTPUT_BLKS_PER_STEP; \
                dCv4_x_valid[i]  = (dCv4_idx[i] / out_hw) < in_num; \
            } \
        }

#endif

//////////////////////////////////////////////////////
// bias macros
//////////////////////////////////////////////////////

#define LOAD_BIAS_V4(_has_bias, _bias, _bias_v4_off) \
        { \
            if( _has_bias && dCv4_y_valid ) \
                Rv4[_bias_v4_off + 0] = ((int4 *) _bias) [grp_id * num_flt_per_grp_pad_v8 + dCv4_idy]; \
        }

#define ADD_BIAS_V4(_has_bias, _bias_v1_off) \
        { \
            if( _has_bias ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
                { \
                    _Pragma("unroll") \
	                for(int j = 0; j < _INT4_TO_4INT_; j++) \
                    { \
                        HADD2_INST(R[i * _INT4_TO_4INT_ + j], R[_bias_v1_off + j], R[i * _INT4_TO_4INT_ + j]); \
	                } \
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
                for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
                { \
                    _Pragma("unroll") \
	                for(int j = 0; j < _INT4_TO_4INT_; j++) \
                    { \
                        HMAX2_INST(R[i * _INT4_TO_4INT_ + j], R[i * _INT4_TO_4INT_ + j], 0, R[i * _INT4_TO_4INT_ + j]); \
	                } \
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
                int * _r_clip_max = (int *) &_clip_max; \
                int * _r_clip_min = (int *) &_clip_min; \
                \
                _Pragma("unroll") \
                for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
                { \
                    _Pragma("unroll") \
	                for(int j = 0; j < _INT4_TO_4INT_; j++) \
                    { \
                        HMIN2_INST(R[i * _INT4_TO_4INT_ + j], R[i * _INT4_TO_4INT_ + j], _r_clip_max[0], R[i * _INT4_TO_4INT_ + j]); \
                        HMAX2_INST(R[i * _INT4_TO_4INT_ + j], R[i * _INT4_TO_4INT_ + j], _r_clip_min[0], R[i * _INT4_TO_4INT_ + j]); \
	                } \
		        } \
		    } \
        }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define LOAD_ELT_V4(_has_elt, _pre_data, _elt_v4_off) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
            { \
       	        if( _has_elt && dCv4_y_valid && dCv4_x_valid[i] ) \
                    Rv4[_elt_v4_off + i] = ((int4 *) _pre_data) [dCv4_base + dCv4_idx[i] * num_flt_per_grp_pad_v8 * num_grp]; \
            } \
        }

#define FUSE_ELT_V4(_has_elt, _elt_v1_off) \
        { \
	        if(_has_elt) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
                { \
                    _Pragma("unroll") \
	                for(int j = 0; j < _INT4_TO_4INT_; j++) { \
                        HADD2_INST(R[i * _INT4_TO_4INT_ + j], R[_elt_v1_off + i * _INT4_TO_4INT_ + j], R[i * _INT4_TO_4INT_ + j]); \
	                } \
	            } \
	        } \
        }

//////////////////////////////////////////////////////
// concat macros
//////////////////////////////////////////////////////

#define SET_CONCAT_OFF_V4(_has_concat, _concat_v4_off) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
            { \
	            _concat_v4_off[i] = (_has_concat) ? dCv4_idx[i] * concat_stride_v8 + concat_offset_v8 : dCv4_idx[i] * num_flt_per_grp_pad_v8 * num_grp; \
	        } \
        }
        
