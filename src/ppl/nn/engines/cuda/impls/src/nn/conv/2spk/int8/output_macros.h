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

#if defined(ENABLE_FUSE)

#define PACK_V4(_R, _R_off) \
        { \
            asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;\n" : "=r"(_R[_R_off + 2]) : "r"(_R[_R_off + 3]), "r"(_R[_R_off + 2]) ); \
            asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;\n": "=r"(_R[_R_off + 0]) : "r"(_R[_R_off + 1]), "r"(_R[_R_off + 0]), "r"(_R[_R_off + 2])); \
        }

#define OUTPUT_BY_INT8_V4(_R) \
        { \
            if( dCv4XValid && dCv4YValid ) \
                ((int*) dC)[concatV4_off + dCv4_off] = _R[0]; \
        }

#elif defined(ENABLE_SPLITK) || defined(ENABLE_SPLITF)

#define OUTPUT_BY_INT4_V1(_Rv4) \
        { \
            if( dCv4XValid && dCv4YValid ) \
            { \
                ((int4 *)dC)[ dCv4_off ] = _Rv4[0]; \
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
                _deScaleV4 = ((float4 *) _dFltScale)[grp_id * numFltPerGrpPadV4 + dCv4_idx]; \
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
            if(_hasBias && dCv4XValid && dCv4YValid) \
            { \
                int4 _biasV4 = ((int4 *)_bias)[grp_id * numFltPerGrpPadV4 + dCv4_idx]; \
	            float* _fBias = (float *) &_biasV4; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                { \
	                fR[i] = fR[i] + _fBias[i]; \
	            } \
            } \
        }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define FUSE_RELU_V4(_hasRelu) \
        { \
	        if(_hasRelu && dCv4XValid && dCv4YValid) \
            { \
                if (_hasRelu == 1) \
                {  \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	                    fR[i] = Max(fR[i], 0); \
	                } \
                } \
		    } \
        }

#if 0
#define FUSE_RELU_V4(_hasRelu) \
        { \
	        if(_hasRelu && dCv4XValid && dCv4YValid) \
            { \
                if (_hasRelu == 1) {  \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	                    fR[i] = Max(fR[i], 0); \
	                } \
                } \
                else if( _hasRelu == 2) { \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
			            fR[i] = __expf(fR[i]) / (1.f + __expf(fR[i])); \
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
	        if(_hasClip && dCv4XValid && dCv4YValid) \
            { \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                { \
			        fR[i] = Min(fR[i], _clipMax); \
			        fR[i] = Max(fR[i], _clipMin); \
	            } \
		    } \
        }

//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define FUSE_PRELU_V4(_hasPrelu, _prelu, _leaky) \
        { \
            if (_hasPrelu && dCv4XValid && dCv4YValid) \
            { \
                if (_hasPrelu == 1) \
                { \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	    		        if(fR[i] < _FLOAT_ZERO_) \
                            fR[i] *= _leaky; \
                    } \
                } \
                \
                if (_hasPrelu == 2) \
                { \
                    int4 _scaleV4  = ((int4 *)_prelu)[grp_id * numFltPerGrpPadV4 + dCv4_idx]; \
                    float * _fScale = (float *) &_scaleV4; \
                    \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	    		        if(fR[i] < _FLOAT_ZERO_) \
                            fR[i] *= _fScale[i]; \
                    } \
                } \
                \
                if (_hasPrelu == 3) \
                { \
                    int4 _scaleV4  = ((int4 *)_prelu)[dCv4_off]; \
                    float * _fScale = (float *) &_scaleV4; \
                    \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	    		        if(fR[i] < _FLOAT_ZERO_) \
                            fR[i] *= _fScale[i]; \
                    } \
                } \
            } \
        }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define FUSE_ELT_V4(_hasElt, _preData) \
        { \
	        if( _hasElt && dCv4XValid && dCv4YValid) \
            { \
	            int  _eltV4 = ((int *) _preData)[dCv4_off]; \
	            int8_t *_eltV1 = (int8_t *) &_eltV4; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                { \
			        fR[i] += (int)_eltV1[i] * preScale; \
	            } \
	        } \
        }

#define SET_CONCAT_OFF_V4(_hasConcat, _concatV4_off) \
        { \
            if (_hasConcat && dCv4XValid && dCv4YValid) \
            { \
                dCv4_off = concatOffsetV4 + dCv4_idy * concatStrideV4 + dCv4_base + dCv4_idx; \
            } \
        }
        
