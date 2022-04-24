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
// pack output macros
//////////////////////////////////////////////////////

#define AND_BIT8_V1(_C, _Cv1_off) \
        { \
            _C[_Cv1_off] = _C[_Cv1_off] & 0xff; \
        }

#define AND_BIT8_V2(_C, _Cv1_off) \
        { \
            _C[_Cv1_off + 0] = _C[_Cv1_off + 0] & 0xff; \
            _C[_Cv1_off + 1] = _C[_Cv1_off + 1] & 0xff; \
        }

#define AND_BIT8_V4(_C, _Cv1_off) \
        { \
            _C[_Cv1_off + 0] = _C[_Cv1_off + 0] & 0xff; \
            _C[_Cv1_off + 1] = _C[_Cv1_off + 1] & 0xff; \
            _C[_Cv1_off + 2] = _C[_Cv1_off + 2] & 0xff; \
            _C[_Cv1_off + 3] = _C[_Cv1_off + 3] & 0xff; \
        }

//////////////////////////////////////////////////////
// half output interface
//////////////////////////////////////////////////////

#define PACK_V2(_C, _Cv1_off) \
        { \
            asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;\n" : "=r"(_C[_Cv1_off + 0]) : "r"(_C[_Cv1_off + 1]), "r"(_C[_Cv1_off + 0]) ); \
        }

#define PACK_V4(_C, _Cv1_off) \
        { \
            asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;\n" : "=r"(_C[_Cv1_off + 2]) : "r"(_C[_Cv1_off + 3]), "r"(_C[_Cv1_off + 2]) ); \
            asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;\n": "=r"(_C[_Cv1_off + 0]) : "r"(_C[_Cv1_off + 1]), "r"(_C[_Cv1_off + 0]), "r"(_C[_Cv1_off + 2])); \
        }

#define OUTPUT_BY_HALF_X2() \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv2YValid[0] && dCv2XValid[i] ) \
                { \
                    AND_BIT8_V2(C, (Cv2_off + i) * _INT2_TO_2INT_); \
                    PACK_V2(C, (Cv2_off + i) * _INT2_TO_2INT_); \
                    dCvHalf[concatV2_off0 + dCv2_idy[0] * numFltV2 + dCv2_idx[i]] = CvHalf[(Cv2_off + i) * _INT2_TO_4HALF_]; \
                } \
                \
                if( dCv2YValid[1] && dCv2XValid[i] ) \
                { \
                    AND_BIT8_V2(C, (Cv2_off + i + NUM_N_STEPS) * _INT2_TO_2INT_); \
                    PACK_V2(C, (Cv2_off + i + NUM_N_STEPS) * _INT2_TO_2INT_); \
                    dCvHalf[concatV2_off1 + dCv2_idy[1] * numFltV2 + dCv2_idx[i]] = CvHalf[(Cv2_off + i + NUM_N_STEPS) * _INT2_TO_4HALF_]; \
                } \
            } \
            \
            dCv2_idy[0]  += TILE_M_PER_STEP; \
            dCv2_idy[1]  += TILE_M_PER_STEP; \
            dCv2YValid[0] = (dCv2_idy[0] < outNHW); \
            dCv2YValid[1] = (dCv2_idy[1] < outNHW); \
        }

//////////////////////////////////////////////////////
// quant interface
//////////////////////////////////////////////////////

#define GET_DEQUANTSCALE_V2(_deScaleV2, _dFltScale, _inScale) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv2YValid[0] && dCv2XValid[i] ) { \
                    _deScaleV2[i] = ((float2 *)_dFltScale) [dCv2_idx[i]]; \
                    _deScaleV2[i].x *= _inScale; \
                    _deScaleV2[i].y *= _inScale; \
                } \
                if( dCv2YValid[1] && dCv2XValid[i] ) { \
                    _deScaleV2[i + NUM_N_STEPS] = ((float2 *)_dFltScale) [dCv2_idx[i + NUM_N_STEPS]]; \
                    _deScaleV2[i + NUM_N_STEPS].x *= _inScale; \
                    _deScaleV2[i + NUM_N_STEPS].y *= _inScale; \
                } \
	        } \
        }

#define DEQUANT_V2(_fCv2, _Cv2, _deScaleV2) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv2YValid[0] && dCv2XValid[i] ) { \
                    _fCv2[Cv2_off + i].x = _Cv2[Cv2_off + i].x * _deScaleV2[i].x; \
                    _fCv2[Cv2_off + i].y = _Cv2[Cv2_off + i].y * _deScaleV2[i].y; \
                } \
                if( dCv2YValid[1] && dCv2XValid[i] ) { \
                    _fCv2[Cv2_off + i + NUM_N_STEPS].x = _Cv2[Cv2_off + i + NUM_N_STEPS].x * _deScaleV2[i].x; \
                    _fCv2[Cv2_off + i + NUM_N_STEPS].y = _Cv2[Cv2_off + i + NUM_N_STEPS].y * _deScaleV2[i].y; \
                } \
	        } \
        }

#define QUANT_V2(_Cv2, _fCv2, _quantScale) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv2YValid[0] && dCv2XValid[i] ) { \
                    _Cv2[Cv2_off + i].x = __float2int_rn(_fCv2[Cv2_off + i].x * _quantScale); \
                    _Cv2[Cv2_off + i].y = __float2int_rn(_fCv2[Cv2_off + i].y * _quantScale); \
                } \
                if( dCv2YValid[1] && dCv2XValid[i] ) { \
                    _Cv2[Cv2_off + i + NUM_N_STEPS].x = __float2int_rn(_fCv2[Cv2_off + i + NUM_N_STEPS].x * _quantScale); \
                    _Cv2[Cv2_off + i + NUM_N_STEPS].y = __float2int_rn(_fCv2[Cv2_off + i + NUM_N_STEPS].y * _quantScale); \
                } \
	        } \
        }

//////////////////////////////////////////////////////
// bias macros
//////////////////////////////////////////////////////

#define ADD_BIAS_V2(_hasBias, _bias) \
        { \
            if( _hasBias ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2YValid[0] && dCv2XValid[i] ) { \
                        float2 f2Bias = ((float2 *)_bias) [dCv2_idx[i]]; \
                        fCv2[Cv2_off + i].x += f2Bias.x; \
                        fCv2[Cv2_off + i].y += f2Bias.y; \
                    } \
                    if( dCv2YValid[1] && dCv2XValid[i] ) { \
                        float2 f2Bias = ((float2 *)_bias) [dCv2_idx[i]]; \
                        fCv2[Cv2_off + i + NUM_N_STEPS].x += f2Bias.x; \
                        fCv2[Cv2_off + i + NUM_N_STEPS].y += f2Bias.y; \
                    } \
                } \
            } \
        }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define FUSE_RELU_V2(_hasRelu) \
        { \
	        if( _hasRelu == 1) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2YValid[0] && dCv2XValid[i] ) { \
                        fCv2[Cv2_off + i].x = Max(fCv2[Cv2_off + i].x, _FLOAT_ZERO_); \
                        fCv2[Cv2_off + i].y = Max(fCv2[Cv2_off + i].y, _FLOAT_ZERO_); \
                    } \
                    if( dCv2YValid[1] && dCv2XValid[i] ) { \
                        fCv2[Cv2_off + i + NUM_N_STEPS].x = Max(fCv2[Cv2_off + i + NUM_N_STEPS].x, _FLOAT_ZERO_); \
                        fCv2[Cv2_off + i + NUM_N_STEPS].y = Max(fCv2[Cv2_off + i + NUM_N_STEPS].y, _FLOAT_ZERO_); \
                    } \
	            } \
	        } \
        }

#if 0
#define FUSE_RELU_V2(_hasRelu) \
        { \
	        if( _hasRelu == 1) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2YValid[0] && dCv2XValid[i] ) { \
                        fCv2[Cv2_off + i].x = Max(fCv2[Cv2_off + i].x, _FLOAT_ZERO_); \
                        fCv2[Cv2_off + i].y = Max(fCv2[Cv2_off + i].y, _FLOAT_ZERO_); \
                    } \
                    if( dCv2YValid[1] && dCv2XValid[i] ) { \
                        fCv2[Cv2_off + i + NUM_N_STEPS].x = Max(fCv2[Cv2_off + i + NUM_N_STEPS].x, _FLOAT_ZERO_); \
                        fCv2[Cv2_off + i + NUM_N_STEPS].y = Max(fCv2[Cv2_off + i + NUM_N_STEPS].y, _FLOAT_ZERO_); \
                    } \
	            } \
	        } \
            else if( _hasRelu == 2) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2YValid[0] && dCv2XValid[i] ) { \
                        fCv2[Cv2_off + i].x = __expf(fCv2[Cv2_off + i].x) / (_FLOAT_ONE_ + __expf(fCv2[Cv2_off + i].x)); \
                        fCv2[Cv2_off + i].y = __expf(fCv2[Cv2_off + i].y) / (_FLOAT_ONE_ + __expf(fCv2[Cv2_off + i].y)); \
                    } \
                    if( dCv2YValid[1] && dCv2XValid[i] ) { \
                        fCv2[Cv2_off + i + NUM_N_STEPS].x = __expf(fCv2[Cv2_off + i + NUM_N_STEPS].x) / (_FLOAT_ONE_ + __expf(fCv2[Cv2_off + i + NUM_N_STEPS].x)); \
                        fCv2[Cv2_off + i + NUM_N_STEPS].y = __expf(fCv2[Cv2_off + i + NUM_N_STEPS].y) / (_FLOAT_ONE_ + __expf(fCv2[Cv2_off + i + NUM_N_STEPS].y)); \
                    } \
	            } \
	        } \
        }
#endif

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define FUSE_CLIP_V2(_hasClip, _clipMax, _clipMin) \
        { \
	        if( _hasClip ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2YValid[0] && dCv2XValid[i] ) { \
                        fCv2[Cv2_off + i].x = Min(fCv2[Cv2_off + i].x, _clipMax); \
                        fCv2[Cv2_off + i].y = Min(fCv2[Cv2_off + i].y, _clipMax); \
                        fCv2[Cv2_off + i].x = Max(fCv2[Cv2_off + i].x, _clipMin); \
                        fCv2[Cv2_off + i].y = Max(fCv2[Cv2_off + i].y, _clipMin); \
                    } \
                    if( dCv2YValid[1] && dCv2XValid[i] ) { \
                        fCv2[Cv2_off + i + NUM_N_STEPS].x = Min(fCv2[Cv2_off + i + NUM_N_STEPS].x, _clipMax); \
                        fCv2[Cv2_off + i + NUM_N_STEPS].y = Min(fCv2[Cv2_off + i + NUM_N_STEPS].y, _clipMax); \
                        fCv2[Cv2_off + i + NUM_N_STEPS].x = Max(fCv2[Cv2_off + i + NUM_N_STEPS].x, _clipMin); \
                        fCv2[Cv2_off + i + NUM_N_STEPS].y = Max(fCv2[Cv2_off + i + NUM_N_STEPS].y, _clipMin); \
                    } \
	            } \
	        } \
        }

//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define FUSE_PRELU_V2(_hasPrelu, _prelu, _leaky) \
        { \
	        if( _hasPrelu == 1) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2YValid[0] && dCv2XValid[i] && fCv2[Cv2_off + i].x < _FLOAT_ZERO_) \
                        fCv2[Cv2_off + i].x *= _leaky; \
                    if( dCv2YValid[0] && dCv2XValid[i] && fCv2[Cv2_off + i].y < _FLOAT_ZERO_) \
                        fCv2[Cv2_off + i].y *= _leaky; \
                    \
                    if( dCv2YValid[1] && dCv2XValid[i] && fCv2[Cv2_off + i + NUM_N_STEPS].x < _FLOAT_ZERO_) \
                        fCv2[Cv2_off + i + NUM_N_STEPS].x *= _leaky; \
                    if( dCv2YValid[1] && dCv2XValid[i] && fCv2[Cv2_off + i + NUM_N_STEPS].y < _FLOAT_ZERO_) \
                        fCv2[Cv2_off + i + NUM_N_STEPS].y *= _leaky; \
	            } \
	        } \
            \
	        if( _hasPrelu == 2) \
            { \
                int2 _scaleV2[NUM_N_STEPS]; \
                float * _fScale = (float *) _scaleV2;\
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                    _scaleV2[i] = dCv2XValid[i] ? ((int2 *)_prelu)[dCv2_idx[i]] : {0, 0}; \
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2YValid[0] && dCv2XValid[i] && fCv2[Cv2_off + i].x < _FLOAT_ZERO_ ) \
                        fCv2[Cv2_off + i].x *= _fScale[i * _INT2_TO_2INT_ + 0]; \
                    if( dCv2YValid[0] && dCv2XValid[i] && fCv2[Cv2_off + i].y < _FLOAT_ZERO_ ) \
                        fCv2[Cv2_off + i].y *= _fScale[i * _INT2_TO_2INT_ + 1]; \
                    \
                    if( dCv2YValid[1] && dCv2XValid[i] && fCv2[Cv2_off + i + NUM_N_STEPS].x < _FLOAT_ZERO_ ) \
                        fCv2[Cv2_off + i + NUM_N_STEPS].x *= _fScale[i * _INT2_TO_2INT_ + 0]; \
                    if( dCv2YValid[1] && dCv2XValid[i] && fCv2[Cv2_off + i + NUM_N_STEPS].y < _FLOAT_ZERO_ ) \
                        fCv2[Cv2_off + i + NUM_N_STEPS].y *= _fScale[i * _INT2_TO_2INT_ + 1]; \
	            } \
	        } \
	        if( _hasPrelu == 3) \
            { \
                int2 _scaleV2[BLK_M_PER_MMA * NUM_N_STEPS]; \
                float * _fScale = (float *) _scaleV2;\
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    _scaleV2[i * BLK_M_PER_MMA + 0] = (dCv2YValid[0] && dCv2XValid[i]) ? ((int2 *)_prelu)[dCv2_idy[0] * numFltV2 + dCv2_idx[i]] : {0, 0}; \
                    _scaleV2[i * BLK_M_PER_MMA + 1] = (dCv2YValid[1] && dCv2XValid[i]) ? ((int2 *)_prelu)[dCv2_idy[1] * numFltV2 + dCv2_idx[i]] : {0, 0}; \
                } \
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2YValid[0] && dCv2XValid[i] && fCv2[Cv2_off + i].x < _FLOAT_ZERO_ ) \
                        fCv2[Cv2_off + i].x *= _fScale[(i * BLK_M_PER_MMA + 0) * _INT2_TO_2INT_ + 0]; \
                    if( dCv2YValid[0] && dCv2XValid[i] && fCv2[Cv2_off + i].y < _FLOAT_ZERO_ ) \
                        fCv2[Cv2_off + i].y *= _fScale[(i * BLK_M_PER_MMA + 0) * _INT2_TO_2INT_ + 1]; \
                    \
                    if( dCv2YValid[1] && dCv2XValid[i] && fCv2[Cv2_off + i + NUM_N_STEPS].x < _FLOAT_ZERO_ ) \
                        fCv2[Cv2_off + i + NUM_N_STEPS].x *= _fScale[(i * BLK_M_PER_MMA + 1) * _INT2_TO_2INT_ + 0]; \
                    if( dCv2YValid[1] && dCv2XValid[i] && fCv2[Cv2_off + i + NUM_N_STEPS].y < _FLOAT_ZERO_ ) \
                        fCv2[Cv2_off + i + NUM_N_STEPS].y *= _fScale[(i * BLK_M_PER_MMA + 1) * _INT2_TO_2INT_ + 1]; \
	            } \
	        } \
    }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define FUSE_ELT_V2(_hasElt, _preData) \
        { \
	        if( _hasElt ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if(dCv2YValid[0] && dCv2XValid[i] ) { \
                        int16_t  _eltV2 = ((int16_t*) _preData) [dCv2_idy[0] * numFltV2 + dCv2_idx[i]]; \
                        int8_t * _eltV1 = (int8_t *) &_eltV2; \
                        \
                        fCv2[Cv2_off + i].x += (int)_eltV1[0] * preScale; \
                        fCv2[Cv2_off + i].y += (int)_eltV1[1] * preScale; \
                    } \
                    if(dCv2YValid[1] && dCv2XValid[i] ) { \
                        int16_t  _eltV2 = ((int16_t*) _preData) [dCv2_idy[1] * numFltV2 + dCv2_idx[i]]; \
                        int8_t * _eltV1 = (int8_t *) &_eltV2; \
                        \
                        fCv2[Cv2_off + i + NUM_N_STEPS].x += (int)_eltV1[0] * preScale; \
                        fCv2[Cv2_off + i + NUM_N_STEPS].y += (int)_eltV1[1] * preScale; \
                    } \
	            } \
	        } \
        }

//////////////////////////////////////////////////////
// concat macros
//////////////////////////////////////////////////////

#define SET_CONCAT_OFF_V2(_has_concat, _concatV2_off0, _concatV2_off1) \
        { \
            _concatV2_off0 = dCv2_idy[0] * numFltV2; \
            _concatV2_off1 = dCv2_idy[1] * numFltV2; \
            if (_has_concat) { \
                if (dCv2YValid[0]) \
                    _concatV2_off0 = concatOffsetV4 * _INT4_TO_8HALF_ + dCv2_idy[0] * concatStrideV4 * _INT4_TO_8HALF_; \
                if (dCv2YValid[1]) \
                    _concatV2_off1 = concatOffsetV4 * _INT4_TO_8HALF_ + dCv2_idy[1] * concatStrideV4 * _INT4_TO_8HALF_; \
            } \
        }

