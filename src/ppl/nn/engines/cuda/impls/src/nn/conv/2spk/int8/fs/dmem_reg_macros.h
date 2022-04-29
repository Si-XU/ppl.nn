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

/////////////////////////////////////////////////////
// common load global memory macros
/////////////////////////////////////////////////////

////////////////////////////////////////
// load dB macros
////////////////////////////////////////

#define LOAD_dBv4_SIZE_16TH(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE_8TH(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE_QTR(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE_HALF(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE1(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE2(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            _regB[1] = ( _fltNValid[1] && _fltCv16Valid ) ? _dB[ _dBv4_off[1] ] : ZEROv4;\
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
            _dBv4_off[1] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE4(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            _regB[1] = ( _fltNValid[1] && _fltCv16Valid ) ? _dB[ _dBv4_off[1] ] : ZEROv4;\
            _regB[2] = ( _fltNValid[2] && _fltCv16Valid ) ? _dB[ _dBv4_off[2] ] : ZEROv4;\
            _regB[3] = ( _fltNValid[3] && _fltCv16Valid ) ? _dB[ _dBv4_off[3] ] : ZEROv4;\
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
            _dBv4_off[1] += TILE_K_V16_PER_CTA; \
            _dBv4_off[2] += TILE_K_V16_PER_CTA; \
            _dBv4_off[3] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE8(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            _regB[1] = ( _fltNValid[1] && _fltCv16Valid ) ? _dB[ _dBv4_off[1] ] : ZEROv4;\
            _regB[2] = ( _fltNValid[2] && _fltCv16Valid ) ? _dB[ _dBv4_off[2] ] : ZEROv4;\
            _regB[3] = ( _fltNValid[3] && _fltCv16Valid ) ? _dB[ _dBv4_off[3] ] : ZEROv4;\
            _regB[4] = ( _fltNValid[4] && _fltCv16Valid ) ? _dB[ _dBv4_off[4] ] : ZEROv4;\
            _regB[5] = ( _fltNValid[5] && _fltCv16Valid ) ? _dB[ _dBv4_off[5] ] : ZEROv4;\
            _regB[6] = ( _fltNValid[6] && _fltCv16Valid ) ? _dB[ _dBv4_off[6] ] : ZEROv4;\
            _regB[7] = ( _fltNValid[7] && _fltCv16Valid ) ? _dB[ _dBv4_off[7] ] : ZEROv4;\
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
            _dBv4_off[1] += TILE_K_V16_PER_CTA; \
            _dBv4_off[2] += TILE_K_V16_PER_CTA; \
            _dBv4_off[3] += TILE_K_V16_PER_CTA; \
            _dBv4_off[4] += TILE_K_V16_PER_CTA; \
            _dBv4_off[5] += TILE_K_V16_PER_CTA; \
            _dBv4_off[6] += TILE_K_V16_PER_CTA; \
            _dBv4_off[7] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE16(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _regB[0]  = ( _fltNValid[0]  && _fltCv16Valid ) ? _dB[ _dBv4_off[0]  ] : ZEROv4;\
            _regB[1]  = ( _fltNValid[1]  && _fltCv16Valid ) ? _dB[ _dBv4_off[1]  ] : ZEROv4;\
            _regB[2]  = ( _fltNValid[2]  && _fltCv16Valid ) ? _dB[ _dBv4_off[2]  ] : ZEROv4;\
            _regB[3]  = ( _fltNValid[3]  && _fltCv16Valid ) ? _dB[ _dBv4_off[3]  ] : ZEROv4;\
            _regB[4]  = ( _fltNValid[4]  && _fltCv16Valid ) ? _dB[ _dBv4_off[4]  ] : ZEROv4;\
            _regB[5]  = ( _fltNValid[5]  && _fltCv16Valid ) ? _dB[ _dBv4_off[5]  ] : ZEROv4;\
            _regB[6]  = ( _fltNValid[6]  && _fltCv16Valid ) ? _dB[ _dBv4_off[6]  ] : ZEROv4;\
            _regB[7]  = ( _fltNValid[7]  && _fltCv16Valid ) ? _dB[ _dBv4_off[7]  ] : ZEROv4;\
            _regB[8]  = ( _fltNValid[8]  && _fltCv16Valid ) ? _dB[ _dBv4_off[8]  ] : ZEROv4;\
            _regB[9]  = ( _fltNValid[9]  && _fltCv16Valid ) ? _dB[ _dBv4_off[9]  ] : ZEROv4;\
            _regB[10] = ( _fltNValid[10] && _fltCv16Valid ) ? _dB[ _dBv4_off[10] ] : ZEROv4;\
            _regB[11] = ( _fltNValid[11] && _fltCv16Valid ) ? _dB[ _dBv4_off[11] ] : ZEROv4;\
            _regB[12] = ( _fltNValid[12] && _fltCv16Valid ) ? _dB[ _dBv4_off[12] ] : ZEROv4;\
            _regB[13] = ( _fltNValid[13] && _fltCv16Valid ) ? _dB[ _dBv4_off[13] ] : ZEROv4;\
            _regB[14] = ( _fltNValid[14] && _fltCv16Valid ) ? _dB[ _dBv4_off[14] ] : ZEROv4;\
            _regB[15] = ( _fltNValid[15] && _fltCv16Valid ) ? _dB[ _dBv4_off[15] ] : ZEROv4;\
            \
            _dBv4_off[0]  += TILE_K_V16_PER_CTA; \
            _dBv4_off[1]  += TILE_K_V16_PER_CTA; \
            _dBv4_off[2]  += TILE_K_V16_PER_CTA; \
            _dBv4_off[3]  += TILE_K_V16_PER_CTA; \
            _dBv4_off[4]  += TILE_K_V16_PER_CTA; \
            _dBv4_off[5]  += TILE_K_V16_PER_CTA; \
            _dBv4_off[6]  += TILE_K_V16_PER_CTA; \
            _dBv4_off[7]  += TILE_K_V16_PER_CTA; \
            _dBv4_off[8]  += TILE_K_V16_PER_CTA; \
            _dBv4_off[9]  += TILE_K_V16_PER_CTA; \
            _dBv4_off[10] += TILE_K_V16_PER_CTA; \
            _dBv4_off[11] += TILE_K_V16_PER_CTA; \
            _dBv4_off[12] += TILE_K_V16_PER_CTA; \
            _dBv4_off[13] += TILE_K_V16_PER_CTA; \
            _dBv4_off[14] += TILE_K_V16_PER_CTA; \
            _dBv4_off[15] += TILE_K_V16_PER_CTA; \
        }

#define SET_dBv4_BOUND(_step_id, _dBv4_off, _fltNValid) \
        { \
            int _fltN_id  =  cta_idx  *  TILE_N_PER_CTA + \
                            _step_id  * (TILE_N_PER_CTA / READ_dBv4_STEPS) + \
                             ldg_idy; \
            \
            _fltNValid  =  _fltN_id < numFltPerGrp; \
            \
            _dBv4_off  =   grp_id   * numChlPerGrpPadV16 * fltHW  * numFltPerGrp + \
                          _fltN_id  * numChlPerGrpPadV16 * fltHW  + \
                           spf_id   * numChlPerGrpPadV16 + \
                           fltCv16_id; \
        }

////////////////////////////////////////
// load dA macros
////////////////////////////////////////

#define SET_dAv4_BOUND(_step_id, _dAv4_off, _inHWValid) \
        { \
            int _outNHW_id    =  cta_idy  *  TILE_M_PER_CTA + \
                                _step_id  * (TILE_M_PER_CTA / READ_dAv4_STEPS) + \
                                 ldg_idy; \
            \
            int _outW_id =  (_outNHW_id % outWidth); \
            int _outH_id =  (_outNHW_id / outWidth) % outHeight; \
            \
            int _inN_id  =   _outNHW_id / outHW; \
            int _inH_id  =     _outH_id * strideHeight; \
            int _inW_id  =     _outW_id * strideWidth; \
            \
	        int _fltH_id = spf_id / fltWidth; \
	        int _fltW_id = spf_id % fltWidth; \
            \
            _inH_id =  _inH_id + _fltH_id * holeHeight - padHeight; \
            _inW_id =  _inW_id + _fltW_id * holeWidth - padWidth;  \
            \
            _dAv4_off  =  (_inN_id  * inHW + _inH_id  * inWidth + _inW_id) * numChlPerGrpPadV16 * numGrp + \
                           grp_id   * numChlPerGrpPadV16 + \
                           fltCv16_id; \
            \
            SET_BOUND_FLT1(_inHWValid, _inN_id, _inH_id, _inW_id); \
        }

#define LOAD_dAv4_SIZE_16TH(_regA, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                _regA[0] = ( _inHWValid[0] && _inCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE_8TH(_regA, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                _regA[0] = ( _inHWValid[0] && _inCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE_QTR(_regA, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                _regA[0] = ( _inHWValid[0] && _inCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE_HALF(_regA, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                _regA[0] = ( _inHWValid[0] && _inCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE1(_regA, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            _regA[0] = ( _inHWValid[0] && _inCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE2(_regA, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            _regA[0] = ( _inHWValid[0] && _inCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            _regA[1] = ( _inHWValid[1] && _inCv16Valid ) ? _dA[ _dAv4_off[1] ] : ZEROv4;\
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
            _dAv4_off[1] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE4(_regA, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            _regA[0] = ( _inHWValid[0] && _inCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            _regA[1] = ( _inHWValid[1] && _inCv16Valid ) ? _dA[ _dAv4_off[1] ] : ZEROv4;\
            _regA[2] = ( _inHWValid[2] && _inCv16Valid ) ? _dA[ _dAv4_off[2] ] : ZEROv4;\
            _regA[3] = ( _inHWValid[3] && _inCv16Valid ) ? _dA[ _dAv4_off[3] ] : ZEROv4;\
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
            _dAv4_off[1] += TILE_K_V16_PER_CTA; \
            _dAv4_off[2] += TILE_K_V16_PER_CTA; \
            _dAv4_off[3] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE8(_regA, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            _regA[0] = ( _inHWValid[0] && _inCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            _regA[1] = ( _inHWValid[1] && _inCv16Valid ) ? _dA[ _dAv4_off[1] ] : ZEROv4;\
            _regA[2] = ( _inHWValid[2] && _inCv16Valid ) ? _dA[ _dAv4_off[2] ] : ZEROv4;\
            _regA[3] = ( _inHWValid[3] && _inCv16Valid ) ? _dA[ _dAv4_off[3] ] : ZEROv4;\
            _regA[4] = ( _inHWValid[4] && _inCv16Valid ) ? _dA[ _dAv4_off[4] ] : ZEROv4;\
            _regA[5] = ( _inHWValid[5] && _inCv16Valid ) ? _dA[ _dAv4_off[5] ] : ZEROv4;\
            _regA[6] = ( _inHWValid[6] && _inCv16Valid ) ? _dA[ _dAv4_off[6] ] : ZEROv4;\
            _regA[7] = ( _inHWValid[7] && _inCv16Valid ) ? _dA[ _dAv4_off[7] ] : ZEROv4;\
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
            _dAv4_off[1] += TILE_K_V16_PER_CTA; \
            _dAv4_off[2] += TILE_K_V16_PER_CTA; \
            _dAv4_off[3] += TILE_K_V16_PER_CTA; \
            _dAv4_off[4] += TILE_K_V16_PER_CTA; \
            _dAv4_off[5] += TILE_K_V16_PER_CTA; \
            _dAv4_off[6] += TILE_K_V16_PER_CTA; \
            _dAv4_off[7] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE16(_regA, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            _regA[0]  = ( _inHWValid[0]  && _inCv16Valid ) ? _dA[ _dAv4_off[0]  ] : ZEROv4;\
            _regA[1]  = ( _inHWValid[1]  && _inCv16Valid ) ? _dA[ _dAv4_off[1]  ] : ZEROv4;\
            _regA[2]  = ( _inHWValid[2]  && _inCv16Valid ) ? _dA[ _dAv4_off[2]  ] : ZEROv4;\
            _regA[3]  = ( _inHWValid[3]  && _inCv16Valid ) ? _dA[ _dAv4_off[3]  ] : ZEROv4;\
            _regA[4]  = ( _inHWValid[4]  && _inCv16Valid ) ? _dA[ _dAv4_off[4]  ] : ZEROv4;\
            _regA[5]  = ( _inHWValid[5]  && _inCv16Valid ) ? _dA[ _dAv4_off[5]  ] : ZEROv4;\
            _regA[6]  = ( _inHWValid[6]  && _inCv16Valid ) ? _dA[ _dAv4_off[6]  ] : ZEROv4;\
            _regA[7]  = ( _inHWValid[7]  && _inCv16Valid ) ? _dA[ _dAv4_off[7]  ] : ZEROv4;\
            _regA[8]  = ( _inHWValid[8]  && _inCv16Valid ) ? _dA[ _dAv4_off[8]  ] : ZEROv4;\
            _regA[9]  = ( _inHWValid[9]  && _inCv16Valid ) ? _dA[ _dAv4_off[9]  ] : ZEROv4;\
            _regA[10] = ( _inHWValid[10] && _inCv16Valid ) ? _dA[ _dAv4_off[10] ] : ZEROv4;\
            _regA[11] = ( _inHWValid[11] && _inCv16Valid ) ? _dA[ _dAv4_off[11] ] : ZEROv4;\
            _regA[12] = ( _inHWValid[12] && _inCv16Valid ) ? _dA[ _dAv4_off[12] ] : ZEROv4;\
            _regA[13] = ( _inHWValid[13] && _inCv16Valid ) ? _dA[ _dAv4_off[13] ] : ZEROv4;\
            _regA[14] = ( _inHWValid[14] && _inCv16Valid ) ? _dA[ _dAv4_off[14] ] : ZEROv4;\
            _regA[15] = ( _inHWValid[15] && _inCv16Valid ) ? _dA[ _dAv4_off[15] ] : ZEROv4;\
            \
            _dAv4_off[0]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[1]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[2]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[3]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[4]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[5]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[6]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[7]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[8]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[9]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[10] += TILE_K_V16_PER_CTA; \
            _dAv4_off[11] += TILE_K_V16_PER_CTA; \
            _dAv4_off[12] += TILE_K_V16_PER_CTA; \
            _dAv4_off[13] += TILE_K_V16_PER_CTA; \
            _dAv4_off[14] += TILE_K_V16_PER_CTA; \
            _dAv4_off[15] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE32(_regA, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            _regA[0]  = ( _inHWValid[0]  && _inCv16Valid ) ? _dA[ _dAv4_off[0]  ] : ZEROv4;\
            _regA[1]  = ( _inHWValid[1]  && _inCv16Valid ) ? _dA[ _dAv4_off[1]  ] : ZEROv4;\
            _regA[2]  = ( _inHWValid[2]  && _inCv16Valid ) ? _dA[ _dAv4_off[2]  ] : ZEROv4;\
            _regA[3]  = ( _inHWValid[3]  && _inCv16Valid ) ? _dA[ _dAv4_off[3]  ] : ZEROv4;\
            _regA[4]  = ( _inHWValid[4]  && _inCv16Valid ) ? _dA[ _dAv4_off[4]  ] : ZEROv4;\
            _regA[5]  = ( _inHWValid[5]  && _inCv16Valid ) ? _dA[ _dAv4_off[5]  ] : ZEROv4;\
            _regA[6]  = ( _inHWValid[6]  && _inCv16Valid ) ? _dA[ _dAv4_off[6]  ] : ZEROv4;\
            _regA[7]  = ( _inHWValid[7]  && _inCv16Valid ) ? _dA[ _dAv4_off[7]  ] : ZEROv4;\
            _regA[8]  = ( _inHWValid[8]  && _inCv16Valid ) ? _dA[ _dAv4_off[8]  ] : ZEROv4;\
            _regA[9]  = ( _inHWValid[9]  && _inCv16Valid ) ? _dA[ _dAv4_off[9]  ] : ZEROv4;\
            _regA[10] = ( _inHWValid[10] && _inCv16Valid ) ? _dA[ _dAv4_off[10] ] : ZEROv4;\
            _regA[11] = ( _inHWValid[11] && _inCv16Valid ) ? _dA[ _dAv4_off[11] ] : ZEROv4;\
            _regA[12] = ( _inHWValid[12] && _inCv16Valid ) ? _dA[ _dAv4_off[12] ] : ZEROv4;\
            _regA[13] = ( _inHWValid[13] && _inCv16Valid ) ? _dA[ _dAv4_off[13] ] : ZEROv4;\
            _regA[14] = ( _inHWValid[14] && _inCv16Valid ) ? _dA[ _dAv4_off[14] ] : ZEROv4;\
            _regA[15] = ( _inHWValid[15] && _inCv16Valid ) ? _dA[ _dAv4_off[15] ] : ZEROv4;\
            _regA[16] = ( _inHWValid[16] && _inCv16Valid ) ? _dA[ _dAv4_off[16] ] : ZEROv4;\
            _regA[17] = ( _inHWValid[17] && _inCv16Valid ) ? _dA[ _dAv4_off[17] ] : ZEROv4;\
            _regA[18] = ( _inHWValid[18] && _inCv16Valid ) ? _dA[ _dAv4_off[18] ] : ZEROv4;\
            _regA[19] = ( _inHWValid[19] && _inCv16Valid ) ? _dA[ _dAv4_off[19] ] : ZEROv4;\
            _regA[20] = ( _inHWValid[20] && _inCv16Valid ) ? _dA[ _dAv4_off[20] ] : ZEROv4;\
            _regA[21] = ( _inHWValid[21] && _inCv16Valid ) ? _dA[ _dAv4_off[21] ] : ZEROv4;\
            _regA[22] = ( _inHWValid[22] && _inCv16Valid ) ? _dA[ _dAv4_off[22] ] : ZEROv4;\
            _regA[23] = ( _inHWValid[23] && _inCv16Valid ) ? _dA[ _dAv4_off[23] ] : ZEROv4;\
            _regA[24] = ( _inHWValid[24] && _inCv16Valid ) ? _dA[ _dAv4_off[24] ] : ZEROv4;\
            _regA[25] = ( _inHWValid[25] && _inCv16Valid ) ? _dA[ _dAv4_off[25] ] : ZEROv4;\
            _regA[26] = ( _inHWValid[26] && _inCv16Valid ) ? _dA[ _dAv4_off[26] ] : ZEROv4;\
            _regA[27] = ( _inHWValid[27] && _inCv16Valid ) ? _dA[ _dAv4_off[27] ] : ZEROv4;\
            _regA[28] = ( _inHWValid[28] && _inCv16Valid ) ? _dA[ _dAv4_off[28] ] : ZEROv4;\
            _regA[29] = ( _inHWValid[29] && _inCv16Valid ) ? _dA[ _dAv4_off[29] ] : ZEROv4;\
            _regA[30] = ( _inHWValid[30] && _inCv16Valid ) ? _dA[ _dAv4_off[30] ] : ZEROv4;\
            _regA[31] = ( _inHWValid[31] && _inCv16Valid ) ? _dA[ _dAv4_off[31] ] : ZEROv4;\
            \
            _dAv4_off[0]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[1]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[2]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[3]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[4]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[5]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[6]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[7]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[8]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[9]  += TILE_K_V16_PER_CTA; \
            _dAv4_off[10] += TILE_K_V16_PER_CTA; \
            _dAv4_off[11] += TILE_K_V16_PER_CTA; \
            _dAv4_off[12] += TILE_K_V16_PER_CTA; \
            _dAv4_off[13] += TILE_K_V16_PER_CTA; \
            _dAv4_off[14] += TILE_K_V16_PER_CTA; \
            _dAv4_off[15] += TILE_K_V16_PER_CTA; \
            _dAv4_off[16] += TILE_K_V16_PER_CTA; \
            _dAv4_off[17] += TILE_K_V16_PER_CTA; \
            _dAv4_off[18] += TILE_K_V16_PER_CTA; \
            _dAv4_off[19] += TILE_K_V16_PER_CTA; \
            _dAv4_off[20] += TILE_K_V16_PER_CTA; \
            _dAv4_off[21] += TILE_K_V16_PER_CTA; \
            _dAv4_off[22] += TILE_K_V16_PER_CTA; \
            _dAv4_off[23] += TILE_K_V16_PER_CTA; \
            _dAv4_off[24] += TILE_K_V16_PER_CTA; \
            _dAv4_off[25] += TILE_K_V16_PER_CTA; \
            _dAv4_off[26] += TILE_K_V16_PER_CTA; \
            _dAv4_off[27] += TILE_K_V16_PER_CTA; \
            _dAv4_off[28] += TILE_K_V16_PER_CTA; \
            _dAv4_off[29] += TILE_K_V16_PER_CTA; \
            _dAv4_off[30] += TILE_K_V16_PER_CTA; \
            _dAv4_off[31] += TILE_K_V16_PER_CTA; \
        }
