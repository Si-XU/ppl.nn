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
// output interface
//////////////////////////////////////////////////////

#define AND_BIT8_V4(_R, _R_off) \
        { \
            _R[_R_off + 0] = _R[_R_off + 0] & 0xff; \
            _R[_R_off + 1] = _R[_R_off + 1] & 0xff; \
            _R[_R_off + 2] = _R[_R_off + 2] & 0xff; \
            _R[_R_off + 3] = _R[_R_off + 3] & 0xff; \
        }

#if defined(ENABLE_FUSE)

#define PACK_V4(_R, _R_off) \
        { \
            asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;\n" : "=r"(_R[_R_off + 2]) : "r"(_R[_R_off + 3]), "r"(_R[_R_off + 2]) ); \
            asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;\n": "=r"(_R[_R_off + 0]) : "r"(_R[_R_off + 1]), "r"(_R[_R_off + 0]), "r"(_R[_R_off + 2])); \
        }

#define OUTPUT_BY_INT8_V4(_R) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
            { \
                if( dCv4YValid && dCv4XValid[i] ) \
                { \
                    AND_BIT8_V4(R, i * _INT4_TO_4INT_); \
                    PACK_V4(R, i * _INT4_TO_4INT_); \
                    ((int*) dC)[dCv4_base + concatV4_off[i]] = _R[i * _INT4_TO_4INT_]; \
                } \
                \
                dCv4_idx[i]   +=  OUTPUT_SIZE_X_IN_THD * OUTPUT_BLKS_PER_STEP; \
                dCv4XValid[i]  = (dCv4_idx[i] / outHW) < inNum; \
            } \
        }

#elif defined(ENABLE_SPLITK)

#define OUTPUT_BY_INT4_V1(_Rv4) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
            { \
                if( dCv4YValid && dCv4XValid[i] ) \
                { \
                    dC[dCv4_base + dCv4_idx[i] * numFltPerGrpPadV4 * numGrp] = _Rv4[i]; \
                } \
                \
                dCv4_idx[i]   +=  OUTPUT_SIZE_X_IN_THD * OUTPUT_BLKS_PER_STEP; \
                dCv4XValid[i]  = (dCv4_idx[i] / outHW) < inNum; \
            } \
        }

#endif

//////////////////////////////////////////////////////
// quant interface
//////////////////////////////////////////////////////

#define GET_DEQUANTSCALE(_deScaleV4, _deScale, _dFltScale, _inScale) \
        { \
        	if(dCv4XValid && dCv4YValid) \
            { \
                _deScaleV4 = ((float4 *) _dFltScale)[grp_id * numFltPerGrpPadV4 + dCv4_idy]; \
                \
                _deScale[0] *= _inScale; \
                _deScale[1] *= _inScale; \
                _deScale[2] *= _inScale; \
                _deScale[3] *= _inScale; \
            } \
        }

#define DEQUANT_V4(_fR, _R, _deScale) \
        { \
        	_fR[0] = _R[0] * _deScale[0]; \
        	_fR[1] = _R[1] * _deScale[1]; \
        	_fR[2] = _R[2] * _deScale[2]; \
        	_fR[3] = _R[3] * _deScale[3]; \
        }

#define QUANT_V4(_R, _fR, _quantScale) \
        { \
           _R[0] = __float2int_rn(_fR[0] * _quantScale); \
           _R[1] = __float2int_rn(_fR[1] * _quantScale); \
           _R[2] = __float2int_rn(_fR[2] * _quantScale); \
           _R[3] = __float2int_rn(_fR[3] * _quantScale); \
        }

//////////////////////////////////////////////////////
// bias macros
//////////////////////////////////////////////////////

#define ADD_BIAS_V4(_hasBias, _bias) \
        { \
            if( _hasBias ) \
            { \
                int4 _biasV4 = ((int4 *)_bias)[grp_id * numFltPerGrpPadV4 + dCv4_idy]; \
	            float* _fBias = (float *) &_biasV4; \
                \
                _Pragma("unroll") \
                for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
                { \
                    _Pragma("unroll") \
	                for(int j = 0; j < _INT4_TO_4INT_; j++) \
                    { \
	                    fR[i * _INT4_TO_4INT_ + j] = fR[i * _INT4_TO_4INT_ + j] + _fBias[j]; \
	                } \
                } \
            } \
        }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define FUSE_RELU_V4(_hasRelu) \
        { \
	        if(_hasRelu && dCv4YValid) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
                { \
                    if (_hasRelu == 1) \
                    { \
                        _Pragma("unroll") \
	                    for(int j = 0; j < _INT4_TO_4INT_; j++) \
                        { \
	                        fR[i * _INT4_TO_4INT_ + j] = Max(fR[i * _INT4_TO_4INT_ + j], 0); \
	                    } \
		            } \
		        } \
		    } \
        }


#if 0
#define FUSE_RELU_V4(_hasRelu) \
        { \
	        if(_hasRelu && dCv4YValid) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
                { \
                    if (_hasRelu == 1) { \
                        _Pragma("unroll") \
	                    for(int j = 0; j < _INT4_TO_4INT_; j++) \
                        { \
	                        fR[i * _INT4_TO_4INT_ + j] = Max(fR[i * _INT4_TO_4INT_ + j], 0); \
	                    } \
                    } else if (_hasRelu == 2) { \
                        _Pragma("unroll") \
	                    for(int j = 0; j < _INT4_TO_4INT_; j++) \
                        { \
			                fR[i * _INT4_TO_4INT_ + j] = __expf(fR[i * _INT4_TO_4INT_ + j]) / (1.f + __expf(fR[i * _INT4_TO_4INT_ + j])); \
                        } \
                    } \
		        } \
		    } \
        }
#endif

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define FUSE_CLIP_V4(_hasClip, _clipMax, _clipMin) \
        { \
	        if(_hasClip && dCv4YValid) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
                { \
                    _Pragma("unroll") \
	                for(int j = 0; j < _INT4_TO_4INT_; j++) \
                    { \
			            fR[i * _INT4_TO_4INT_ + j] = Min(fR[i * _INT4_TO_4INT_ + j], _clipMax); \
			            fR[i * _INT4_TO_4INT_ + j] = Max(fR[i * _INT4_TO_4INT_ + j], _clipMin); \
	                } \
		        } \
		    } \
        }

//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define FUSE_PRELU_V4(_hasPrelu, _prelu, _leaky) \
        { \
            if (_hasPrelu && dCv4YValid) { \
                _Pragma("unroll") \
                for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++)  \
                { \
                    if (_hasPrelu == 1) {  \
                        _Pragma("unroll") \
                        for (int j = 0; j < _INT4_TO_4INT_; j++) \
                        { \
	    		            if(fR[i * _INT4_TO_4INT_ + j] < _FLOAT_ZERO_) \
                                fR[i * _INT4_TO_4INT_ + j] *= _leaky; \
                        } \
                    } \
                    \
                    if (_hasPrelu == 2) { \
                        int4 _scaleV4 = ((int4 *)_prelu)[grp_id * numFltPerGrpPadV4 + dCv4_idy]; \
                        float * _fScale = (float *) &_scaleV4; \
                        \
                        _Pragma("unroll") \
                        for (int j = 0; j < _INT4_TO_4INT_; j++) \
                        { \
	    		            if(fR[i * _INT4_TO_4INT_ + j] < _FLOAT_ZERO_) \
                                fR[i * _INT4_TO_4INT_ + j] *= _fScale; \
                        } \
                    } \
                    \
                    if (_hasPrelu == 3) { \
                        int4 _scaleV4[OUTPUT_BLKS_PER_STEP]; \
                        float * _fScale = (float *) &_scaleV4; \
                        \
                        if(dCv4XValid[i]) \
                            _scaleV4[i] = ((int4 *)_prelu)[dCv4_base + dCv4_idx[i] * numFltPerGrpPadV4 * numGrp]; \
                        \
                        _Pragma("unroll") \
                        for (int j = 0; j < _INT4_TO_4INT_; j++) \
                        { \
	    		            if(fR[i * _INT4_TO_4INT_ + j] < _FLOAT_ZERO_) \
                                fR[i * _INT4_TO_4INT_ + j] *= _fScale[i * _INT4_TO_4INT_ + j]; \
                        } \
                    } \
                } \
            } \
        }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define FUSE_ELT_V4(_hasElt, _preData) \
        { \
       	    if(_hasElt && dCv4YValid) \
            { \
                int _eltV4[OUTPUT_BLKS_PER_STEP]; \
                int8_t *_eltV1 = (int8_t *)&_eltV4; \
                \
                _Pragma("unroll") \
                for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
                { \
                    if(dCv4XValid[i]) \
                        _eltV4[i] = ((int *) _preData) [dCv4_base + dCv4_idx[i] * numFltPerGrpPadV4 * numGrp]; \
                    \
                    _Pragma("unroll") \
                    for (int j = 0; j < _INT4_TO_4INT_; j++) \
                    { \
			            fR[i * _INT4_TO_4INT_ + j] += (int)_eltV1[i * _INT4_TO_4INT_ + j] * preScale; \
	                } \
	            } \
	        } \
        }

//////////////////////////////////////////////////////
// concat macros
//////////////////////////////////////////////////////

#define SET_CONCAT_OFF_V4(_hasConcat, _concatV4_off) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
            { \
	            _concatV4_off[i] = (_hasConcat) ? dCv4_idx[i] * concatStrideV4 + concatOffsetV4 : dCv4_idx[i] * numFltPerGrpPadV4 * numGrp; \
	        } \
        }
        