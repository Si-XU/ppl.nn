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

#define OUTPUT_1x1_BY_INT1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv2[dCv1_idx[0] + concat_v1_off0] = outData[0]; \
        }

#define OUTPUT_1x2_BY_INT1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv2[dCv1_idx[0] + concat_v1_off0] = outData[0]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[1]) dCv2[dCv1_idx[1] + concat_v1_off0] = outData[1]; \
        }

#define OUTPUT_1x4_BY_INT1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv2[dCv1_idx[0] + concat_v1_off0] = outData[0]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[1]) dCv2[dCv1_idx[1] + concat_v1_off0] = outData[1]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[2]) dCv2[dCv1_idx[2] + concat_v1_off0] = outData[2]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[3]) dCv2[dCv1_idx[3] + concat_v1_off0] = outData[3]; \
        }

#else

#define OUTPUT_1x1_BY_INT1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] = outData[0]; \
        }

#define OUTPUT_1x2_BY_INT1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] = outData[0]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[1]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] = outData[1]; \
        }

#define OUTPUT_1x4_BY_INT1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] = outData[0]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[1]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] = outData[1]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[2]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]] = outData[2]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[3]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]] = outData[3]; \
        }

#endif

//////////////////////////////////////////////////////
// bias macros
//////////////////////////////////////////////////////

#define ADD_BIAS_1x1_V1(_has_bias, _bias, _step) \
        { \
            if(_has_bias) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[0]]; \
		    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x + f2Bias.x; \
		    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y + f2Bias.y; \
	        } \
            } \
        }

#define ADD_BIAS_1x2_V1(_has_bias, _bias, _step) \
        { \
            if(_has_bias) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[0]]; \
		    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x + f2Bias.x; \
		    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y + f2Bias.y; \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[1]]; \
		    fCv2[Cv1_off + 1].x = fCv2[Cv1_off + 1].x + f2Bias.x; \
		    fCv2[Cv1_off + 1].y = fCv2[Cv1_off + 1].y + f2Bias.y; \
	        } \
            } \
        }

#define ADD_BIAS_1x4_V1(_has_bias, _bias, _step) \
        { \
            if(_has_bias) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[0]]; \
		    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x + f2Bias.x; \
		    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y + f2Bias.y; \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[1]]; \
		    fCv2[Cv1_off + 1].x = fCv2[Cv1_off + 1].x + f2Bias.x; \
		    fCv2[Cv1_off + 1].y = fCv2[Cv1_off + 1].y + f2Bias.y; \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[2]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[2]]; \
		    fCv2[Cv1_off + 2].x = fCv2[Cv1_off + 2].x + f2Bias.x; \
		    fCv2[Cv1_off + 2].y = fCv2[Cv1_off + 2].y + f2Bias.y; \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[3]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[3]]; \
		    fCv2[Cv1_off + 3].x = fCv2[Cv1_off + 3].x + f2Bias.x; \
		    fCv2[Cv1_off + 3].y = fCv2[Cv1_off + 3].y + f2Bias.y; \
	        } \
            } \
        }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define FUSE_RELU_1x1_V1(_has_relu) \
        { \
	        if(_has_relu == 1) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, 0); \
		    fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, 0); \
	        } \
	    } \
            else if(_has_relu == 2) \
            { \
	        float ONE = 1.f; \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = __expf(fCv2[Cv1_off + 0].x) / (ONE + __expf(fCv2[Cv1_off + 0].x)); \
		    fCv2[Cv1_off + 0].y = __expf(fCv2[Cv1_off + 0].y) / (ONE + __expf(fCv2[Cv1_off + 0].y)); \
		} \
	    } \
        }

#define FUSE_RELU_1x2_V1(_has_relu) \
        { \
	        if(_has_relu == 1) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, 0); \
		    fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, 0); \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
		    fCv2[Cv1_off + 1].x = Max(fCv2[Cv1_off + 1].x, 0); \
		    fCv2[Cv1_off + 1].y = Max(fCv2[Cv1_off + 1].y, 0); \
	        } \
	    } \
            else if(_has_relu == 2) \
            { \
	        float ONE = 1.f; \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = __expf(fCv2[Cv1_off + 0].x) / (ONE + __expf(fCv2[Cv1_off + 0].x)); \
		    fCv2[Cv1_off + 0].y = __expf(fCv2[Cv1_off + 0].y) / (ONE + __expf(fCv2[Cv1_off + 0].y)); \
		} \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
		    fCv2[Cv1_off + 1].x = __expf(fCv2[Cv1_off + 1].x) / (ONE + __expf(fCv2[Cv1_off + 1].x)); \
		    fCv2[Cv1_off + 1].y = __expf(fCv2[Cv1_off + 1].y) / (ONE + __expf(fCv2[Cv1_off + 1].y)); \
		} \
	    } \
        } 
#define FUSE_RELU_1x4_V1(_has_relu) \
        { \
	        if(_has_relu == 1) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, 0); \
		    fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, 0); \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
		    fCv2[Cv1_off + 1].x = Max(fCv2[Cv1_off + 1].x, 0); \
		    fCv2[Cv1_off + 1].y = Max(fCv2[Cv1_off + 1].y, 0); \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[2]){ \
		    fCv2[Cv1_off + 2].x = Max(fCv2[Cv1_off + 2].x, 0); \
		    fCv2[Cv1_off + 2].y = Max(fCv2[Cv1_off + 2].y, 0); \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[3]){ \
		    fCv2[Cv1_off + 3].x = Max(fCv2[Cv1_off + 3].x, 0); \
		    fCv2[Cv1_off + 3].y = Max(fCv2[Cv1_off + 3].y, 0); \
	        } \
	    } \
            else if(_has_relu == 2) \
            { \
	        float ONE = 1.f; \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = __expf(fCv2[Cv1_off + 0].x) / (ONE + __expf(fCv2[Cv1_off + 0].x)); \
		    fCv2[Cv1_off + 0].y = __expf(fCv2[Cv1_off + 0].y) / (ONE + __expf(fCv2[Cv1_off + 0].y)); \
		} \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
		    fCv2[Cv1_off + 1].x = __expf(fCv2[Cv1_off + 1].x) / (ONE + __expf(fCv2[Cv1_off + 1].x)); \
		    fCv2[Cv1_off + 1].y = __expf(fCv2[Cv1_off + 1].y) / (ONE + __expf(fCv2[Cv1_off + 1].y)); \
		} \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[2]){ \
		    fCv2[Cv1_off + 2].x = __expf(fCv2[Cv1_off + 2].x) / (ONE + __expf(fCv2[Cv1_off + 2].x)); \
		    fCv2[Cv1_off + 2].y = __expf(fCv2[Cv1_off + 2].y) / (ONE + __expf(fCv2[Cv1_off + 2].y)); \
		} \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[3]){ \
		    fCv2[Cv1_off + 3].x = __expf(fCv2[Cv1_off + 3].x) / (ONE + __expf(fCv2[Cv1_off + 3].x)); \
		    fCv2[Cv1_off + 3].y = __expf(fCv2[Cv1_off + 3].y) / (ONE + __expf(fCv2[Cv1_off + 3].y)); \
		} \
	    } \
        }

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define FUSE_CLIP_1x1_V1(_has_clip, _clip_max, _clip_min) \
        { \
	    if(_has_clip) \
            { \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Min(fCv2[Cv1_off + 0].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Min(fCv2[Cv1_off + 0].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, _clip_min); \
	    } \
        }

#define FUSE_CLIP_1x2_V1(_has_clip, _clip_max, _clip_min) \
        { \
	        if(_has_clip) \
            { \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Min(fCv2[Cv1_off + 0].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Min(fCv2[Cv1_off + 0].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].x = Min(fCv2[Cv1_off + 1].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].y = Min(fCv2[Cv1_off + 1].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].x = Max(fCv2[Cv1_off + 1].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].y = Max(fCv2[Cv1_off + 1].y, _clip_min); \
	    } \
        }

#define FUSE_CLIP_1x4_V1(_has_clip, _clip_max, _clip_min) \
        { \
	        if(_has_clip) \
            { \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Min(fCv2[Cv1_off + 0].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Min(fCv2[Cv1_off + 0].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].x = Min(fCv2[Cv1_off + 1].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].y = Min(fCv2[Cv1_off + 1].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].x = Max(fCv2[Cv1_off + 1].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].y = Max(fCv2[Cv1_off + 1].y, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[2]) fCv2[Cv1_off + 2].x = Min(fCv2[Cv1_off + 2].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[2]) fCv2[Cv1_off + 2].y = Min(fCv2[Cv1_off + 2].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[2]) fCv2[Cv1_off + 2].x = Max(fCv2[Cv1_off + 2].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[2]) fCv2[Cv1_off + 2].y = Max(fCv2[Cv1_off + 2].y, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[3]) fCv2[Cv1_off + 3].x = Min(fCv2[Cv1_off + 3].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[3]) fCv2[Cv1_off + 3].y = Min(fCv2[Cv1_off + 3].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[3]) fCv2[Cv1_off + 3].x = Max(fCv2[Cv1_off + 3].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[3]) fCv2[Cv1_off + 3].y = Max(fCv2[Cv1_off + 3].y, _clip_min); \
	    } \
        } 
//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define FUSE_PRELU_1x1_V1(_has_prelu, _prelu, _leaky) \
        { \
       	    if(_has_prelu == 1 && dCv1_x_valid[0]) \
            { \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \
                { \
                    if(dCv1_y_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _leaky); \
                    \
                    if(dCv1_y_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _leaky); \
	            } \
	        } \
            \
       	    if(_has_prelu == 2 && dCv1_x_valid[0]) \
            { \
	            int      _scale0_v1 = ((int  *) _prelu) [dCv1_idx[0]]; \
	            __half * _hscale0  = (__half *) &_scale0_v1; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \
                { \
                    if(dCv1_y_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]); \
                    \
                    if(dCv1_y_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale0[i]); \
	            } \
	        } \
            \
       	    if(_has_prelu == 3 && dCv1_x_valid[0]) \
            { \
                int      _scale0_v1 = dCv1_y_valid[0] ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
                int      _scale1_v1 = dCv1_y_valid[1] ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] : 0; \
                \
	            __half * _hscale0  = (__half *) &_scale0_v1; \
	            __half * _hscale1  = (__half *) &_scale1_v1; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \
                { \
                    if(dCv1_y_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]); \
                    \
                    if(dCv1_y_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale1[i]); \
	            } \
	        } \
        }

#define FUSE_PRELU_1x2_V1(_has_prelu, _prelu, _leaky) \
        { \
       	    if(_has_prelu == 1) \
            { \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \
                { \
                    if(dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _leaky); \
                    if(dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _leaky); \
                    \
                    if(dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _leaky); \
                    if(dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _leaky); \
	            } \
	        } \
            \
       	    if(_has_prelu == 2) \
            { \
	            int      _scale0_v1 = dCv1_x_valid[0] ? ((int  *) _prelu) [dCv1_idx[0]] : 0; \
	            int      _scale1_v1 = dCv1_x_valid[1] ? ((int  *) _prelu) [dCv1_idx[1]] : 0; \
	            __half * _hscale0  = (__half *) &_scale0_v1; \
	            __half * _hscale1  = (__half *) &_scale1_v1; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \
                { \
                    if(dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]); \
                    if(dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale1[i]); \
                    \
                    if(dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale0[i]); \
                    if(dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale1[i]); \
	            } \
	        } \
            \
       	    if(_has_prelu == 3) \
            { \
                int      _scale00_v1 = (dCv1_y_valid[0] && dCv1_x_valid[0]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
                int      _scale01_v1 = (dCv1_y_valid[0] && dCv1_x_valid[1]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] : 0; \
                \
                int      _scale10_v1 = (dCv1_y_valid[1] && dCv1_x_valid[0]) ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] : 0; \
                int      _scale11_v1 = (dCv1_y_valid[1] && dCv1_x_valid[1]) ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]] : 0; \
                \
	            __half * _hscale00  = (__half *) &_scale00_v1; \
	            __half * _hscale01  = (__half *) &_scale01_v1; \
                \
	            __half * _hscale10  = (__half *) &_scale10_v1; \
	            __half * _hscale11  = (__half *) &_scale11_v1; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \
                { \
                    if(dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale00[i]); \
                    if(dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale01[i]); \
                    \
                    if(dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale10[i]); \
                    if(dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale11[i]); \
	            } \
	        } \
        }

#define FUSE_PRELU_1x4_V1(_has_prelu, _prelu, _leaky) \
        { \
       	    if(_has_prelu == 1) \
            { \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \
                { \
                    if(dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _leaky); \
                    if(dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _leaky); \
                    if(dCv1_y_valid[0] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _leaky); \
                    if(dCv1_y_valid[0] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _leaky); \
                    \
                    if(dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], _leaky); \
                    if(dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], _leaky); \
                    if(dCv1_y_valid[1] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], _leaky); \
                    if(dCv1_y_valid[1] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], _leaky); \
	            } \
	        } \
            \
       	    if(_has_prelu == 2) \
            { \
	            int      _scale0_v1 = dCv1_x_valid[0] ? ((int  *) _prelu) [dCv1_idx[0]] : 0; \
	            int      _scale1_v1 = dCv1_x_valid[1] ? ((int  *) _prelu) [dCv1_idx[1]] : 0; \
	            int      _scale2_v1 = dCv1_x_valid[2] ? ((int  *) _prelu) [dCv1_idx[2]] : 0; \
	            int      _scale3_v1 = dCv1_x_valid[3] ? ((int  *) _prelu) [dCv1_idx[3]] : 0; \
	            __half * _hscale0  = (__half *) &_scale0_v1; \
	            __half * _hscale1  = (__half *) &_scale1_v1; \
	            __half * _hscale2  = (__half *) &_scale2_v1; \
	            __half * _hscale3  = (__half *) &_scale3_v1; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \
                { \
                    if(dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]); \
                    if(dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale1[i]); \
                    if(dCv1_y_valid[0] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale2[i]); \
                    if(dCv1_y_valid[0] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale3[i]); \
                    \
                    if(dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], _hscale0[i]); \
                    if(dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], _hscale1[i]); \
                    if(dCv1_y_valid[1] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], _hscale2[i]); \
                    if(dCv1_y_valid[1] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], _hscale3[i]); \
	            } \
	        } \
            \
       	    if(_has_prelu == 3) \
            { \
                int      _scale00_v1 = (dCv1_y_valid[0] && dCv1_x_valid[0]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
                int      _scale01_v1 = (dCv1_y_valid[0] && dCv1_x_valid[1]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] : 0; \
                int      _scale02_v1 = (dCv1_y_valid[0] && dCv1_x_valid[2]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]] : 0; \
                int      _scale03_v1 = (dCv1_y_valid[0] && dCv1_x_valid[3]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]] : 0; \
                \
                int      _scale10_v1 = (dCv1_y_valid[1] && dCv1_x_valid[0]) ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] : 0; \
                int      _scale11_v1 = (dCv1_y_valid[1] && dCv1_x_valid[1]) ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]] : 0; \
                int      _scale12_v1 = (dCv1_y_valid[1] && dCv1_x_valid[2]) ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[2]] : 0; \
                int      _scale13_v1 = (dCv1_y_valid[1] && dCv1_x_valid[3]) ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[3]] : 0; \
                \
	            __half * _hscale00  = (__half *) &_scale00_v1; \
	            __half * _hscale01  = (__half *) &_scale01_v1; \
	            __half * _hscale02  = (__half *) &_scale02_v1; \
	            __half * _hscale03  = (__half *) &_scale03_v1; \
                \
	            __half * _hscale10  = (__half *) &_scale10_v1; \
	            __half * _hscale11  = (__half *) &_scale11_v1; \
	            __half * _hscale12  = (__half *) &_scale12_v1; \
	            __half * _hscale13  = (__half *) &_scale13_v1; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \
                { \
                    if(dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale00[i]); \
                    if(dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale01[i]); \
                    if(dCv1_y_valid[0] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale02[i]); \
                    if(dCv1_y_valid[0] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale03[i]); \
                    \
                    if(dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], _hscale10[i]); \
                    if(dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], _hscale11[i]); \
                    if(dCv1_y_valid[1] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], _hscale12[i]); \
                    if(dCv1_y_valid[1] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], 0)) \
                        hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], _hscale13[i]); \
	            } \
	        } \
        }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define FUSE_ELT_1x1_V1(_has_elt, _pre_data) \
        { \
	        if(_has_elt) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]]); \
                \
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]]); \
	        } \
        }

#define FUSE_ELT_1x2_V1(_has_elt, _pre_data) \
        { \
	        if(_has_elt) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]]); \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]]); \
                \
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 2] = __hadd2(h2C[Cv1_off + 2], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]]); \
                if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 3] = __hadd2(h2C[Cv1_off + 3], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]]); \
	        } \
        }

#define FUSE_ELT_1x4_V1(_has_elt, _pre_data) \
        { \
	        if(_has_elt) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]]); \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]]); \
                if(dCv1_y_valid[0] && dCv1_x_valid[2]) h2C[Cv1_off + 2] = __hadd2(h2C[Cv1_off + 2], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]]); \
                if(dCv1_y_valid[0] && dCv1_x_valid[3]) h2C[Cv1_off + 3] = __hadd2(h2C[Cv1_off + 3], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]]); \
                \
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 4] = __hadd2(h2C[Cv1_off + 4], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]]); \
                if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 5] = __hadd2(h2C[Cv1_off + 5], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]]); \
                if(dCv1_y_valid[1] && dCv1_x_valid[2]) h2C[Cv1_off + 6] = __hadd2(h2C[Cv1_off + 6], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[2]]); \
                if(dCv1_y_valid[1] && dCv1_x_valid[3]) h2C[Cv1_off + 7] = __hadd2(h2C[Cv1_off + 7], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[3]]); \
	        } \
        }

//////////////////////////////////////////////////////
// concat macros
//////////////////////////////////////////////////////
#define OFF(_off, i)    ( _off#i )

//FIXME _INT4_TO_4HALF2_
#define SET_CONCAT_OFF_V1(_has_concat, _concat_v1_off0) \
        { \
                _concat_v1_off0 = dCv1_idy[0] * num_flt_v2; \
                /*_concat_v1_off1 = dCv1_idy[1] * num_flt_v2;*/ \
		/*for(int b = 0; b < BLK_M_PER_MMA; b++){ \
                    OFF(_concat_v1_off, b) = dCv1_idy[0] * num_flt_v2; \
		}*/ \
	        if(_has_concat) \
            { \
                if(dCv1_y_valid[0]) _concat_v1_off0 = concat_offset_v8 * _INT4_TO_4HALF2_ + dCv1_idy[0] * concat_stride_v8 * _INT4_TO_4HALF2_; \
		/*for(int b = 0; b < BLK_M_PER_MMA; b++){ \
                    if(dCv1_y_valid[b]) OFF(_concat_v1_off, b) = concat_offset_v8 * _INT4_TO_4HALF2_ + dCv1_idy[b] * concat_stride_v8 * _INT4_TO_4HALF2_; \
	        }*/ \
	    } \
        }
