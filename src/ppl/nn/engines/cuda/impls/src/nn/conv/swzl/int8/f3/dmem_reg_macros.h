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
// load dA macros
////////////////////////////////////////

#define LOAD_dAv4_SIZE_16TH(_regA, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                _regA[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE_8TH(_regA, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                _regA[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE_QTR(_regA, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                _regA[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE_HALF(_regA, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                _regA[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE1(_regA, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            _regA[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE2(_regA, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            _dAv4_off[1] += cFltLut.idx[cLut_id]; \
            \
            _regA[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            _regA[1] = ( _fltNValid[1] && _fltCv16Valid ) ? _dA[ _dAv4_off[1] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE4(_regA, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            _dAv4_off[1] += cFltLut.idx[cLut_id]; \
            _dAv4_off[2] += cFltLut.idx[cLut_id]; \
            _dAv4_off[3] += cFltLut.idx[cLut_id]; \
            \
            _regA[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            _regA[1] = ( _fltNValid[1] && _fltCv16Valid ) ? _dA[ _dAv4_off[1] ] : ZEROv4;\
            _regA[2] = ( _fltNValid[2] && _fltCv16Valid ) ? _dA[ _dAv4_off[2] ] : ZEROv4;\
            _regA[3] = ( _fltNValid[3] && _fltCv16Valid ) ? _dA[ _dAv4_off[3] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE8(_regA, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            _dAv4_off[1] += cFltLut.idx[cLut_id]; \
            _dAv4_off[2] += cFltLut.idx[cLut_id]; \
            _dAv4_off[3] += cFltLut.idx[cLut_id]; \
            _dAv4_off[4] += cFltLut.idx[cLut_id]; \
            _dAv4_off[5] += cFltLut.idx[cLut_id]; \
            _dAv4_off[6] += cFltLut.idx[cLut_id]; \
            _dAv4_off[7] += cFltLut.idx[cLut_id]; \
            \
            _regA[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            _regA[1] = ( _fltNValid[1] && _fltCv16Valid ) ? _dA[ _dAv4_off[1] ] : ZEROv4;\
            _regA[2] = ( _fltNValid[2] && _fltCv16Valid ) ? _dA[ _dAv4_off[2] ] : ZEROv4;\
            _regA[3] = ( _fltNValid[3] && _fltCv16Valid ) ? _dA[ _dAv4_off[3] ] : ZEROv4;\
            _regA[4] = ( _fltNValid[4] && _fltCv16Valid ) ? _dA[ _dAv4_off[4] ] : ZEROv4;\
            _regA[5] = ( _fltNValid[5] && _fltCv16Valid ) ? _dA[ _dAv4_off[5] ] : ZEROv4;\
            _regA[6] = ( _fltNValid[6] && _fltCv16Valid ) ? _dA[ _dAv4_off[6] ] : ZEROv4;\
            _regA[7] = ( _fltNValid[7] && _fltCv16Valid ) ? _dA[ _dAv4_off[7] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE16(_regA, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0]  += cFltLut.idx[cLut_id]; \
            _dAv4_off[1]  += cFltLut.idx[cLut_id]; \
            _dAv4_off[2]  += cFltLut.idx[cLut_id]; \
            _dAv4_off[3]  += cFltLut.idx[cLut_id]; \
            _dAv4_off[4]  += cFltLut.idx[cLut_id]; \
            _dAv4_off[5]  += cFltLut.idx[cLut_id]; \
            _dAv4_off[6]  += cFltLut.idx[cLut_id]; \
            _dAv4_off[7]  += cFltLut.idx[cLut_id]; \
            _dAv4_off[8]  += cFltLut.idx[cLut_id]; \
            _dAv4_off[9]  += cFltLut.idx[cLut_id]; \
            _dAv4_off[10] += cFltLut.idx[cLut_id]; \
            _dAv4_off[11] += cFltLut.idx[cLut_id]; \
            _dAv4_off[12] += cFltLut.idx[cLut_id]; \
            _dAv4_off[13] += cFltLut.idx[cLut_id]; \
            _dAv4_off[14] += cFltLut.idx[cLut_id]; \
            _dAv4_off[15] += cFltLut.idx[cLut_id]; \
            \
            _regA[0]  = ( _fltNValid[0]  && _fltCv16Valid ) ? _dA[ _dAv4_off[0]  ] : ZEROv4;\
            _regA[1]  = ( _fltNValid[1]  && _fltCv16Valid ) ? _dA[ _dAv4_off[1]  ] : ZEROv4;\
            _regA[2]  = ( _fltNValid[2]  && _fltCv16Valid ) ? _dA[ _dAv4_off[2]  ] : ZEROv4;\
            _regA[3]  = ( _fltNValid[3]  && _fltCv16Valid ) ? _dA[ _dAv4_off[3]  ] : ZEROv4;\
            _regA[4]  = ( _fltNValid[4]  && _fltCv16Valid ) ? _dA[ _dAv4_off[4]  ] : ZEROv4;\
            _regA[5]  = ( _fltNValid[5]  && _fltCv16Valid ) ? _dA[ _dAv4_off[5]  ] : ZEROv4;\
            _regA[6]  = ( _fltNValid[6]  && _fltCv16Valid ) ? _dA[ _dAv4_off[6]  ] : ZEROv4;\
            _regA[7]  = ( _fltNValid[7]  && _fltCv16Valid ) ? _dA[ _dAv4_off[7]  ] : ZEROv4;\
            _regA[8]  = ( _fltNValid[8]  && _fltCv16Valid ) ? _dA[ _dAv4_off[8]  ] : ZEROv4;\
            _regA[9]  = ( _fltNValid[9]  && _fltCv16Valid ) ? _dA[ _dAv4_off[9]  ] : ZEROv4;\
            _regA[10] = ( _fltNValid[10] && _fltCv16Valid ) ? _dA[ _dAv4_off[10] ] : ZEROv4;\
            _regA[11] = ( _fltNValid[11] && _fltCv16Valid ) ? _dA[ _dAv4_off[11] ] : ZEROv4;\
            _regA[12] = ( _fltNValid[12] && _fltCv16Valid ) ? _dA[ _dAv4_off[12] ] : ZEROv4;\
            _regA[13] = ( _fltNValid[13] && _fltCv16Valid ) ? _dA[ _dAv4_off[13] ] : ZEROv4;\
            _regA[14] = ( _fltNValid[14] && _fltCv16Valid ) ? _dA[ _dAv4_off[14] ] : ZEROv4;\
            _regA[15] = ( _fltNValid[15] && _fltCv16Valid ) ? _dA[ _dAv4_off[15] ] : ZEROv4;\
        }

#define SET_dAv4_BOUND(_step_id, _dAv4_off, _fltNValid) \
        { \
            int _fltN_id  =  cta_idy  *  TILE_M_PER_CTA + \
                            _step_id  * (TILE_M_PER_CTA / READ_dAv4_STEPS) + \
                             ldg_idy; \
            \
            _fltNValid  =  _fltN_id < numFltPerGrp; \
            \
            _dAv4_off  =   grp_id   * fltHW * numChlPerGrpPadV16 * numFltPerGrp + \
                          _fltN_id  * fltHW * numChlPerGrpPadV16 + \
                           fltCv16_id; \
        }

////////////////////////////////////////
// load dB macros
////////////////////////////////////////

#define SET_dBv4_BOUND(_step_id, _dBv4_off, _inHWMask) \
        { \
            int _outNHW_id    =  cta_idx  *  TILE_N_PER_CTA + \
                                _step_id  * (TILE_N_PER_CTA / READ_dBv4_STEPS) + \
                                 ldg_idy; \
            \
            int _outW_id =  (_outNHW_id % outWidth); \
            int _outH_id =  (_outNHW_id / outWidth) % outHeight; \
            \
            int _inN_id  =   _outNHW_id / outHW; \
            int _inH_id  =     _outH_id * strideHeight; \
            int _inW_id  =     _outW_id * strideWidth; \
            \
            _dBv4_off  =  (_inN_id  * inHW + _inH_id  * inWidth + _inW_id) * numChlPerGrpPadV16 * numGrp + \
                           grp_id   * numChlPerGrpPadV16 + \
                           fltCv16_id; \
            \
            _inH_id =  _inH_id - padHeight; \
            _inW_id =  _inW_id - padWidth;  \
            \
            SET_BOUND_FLT3(_inHWMask, _inN_id, _inH_id, _inW_id); \
        }

#define LOAD_dBv4_SIZE_16TH(_regB, _dB, _dBv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                _regB[0] = ( (_fltHW_bid & inHWMask[0]) && _inCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE_8TH(_regB, _dB, _dBv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                _regB[0] = ( (_fltHW_bid & inHWMask[0]) && _inCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE_QTR(_regB, _dB, _dBv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                _regB[0] = ( (_fltHW_bid & inHWMask[0]) && _inCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE_HALF(_regB, _dB, _dBv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                _regB[0] = ( (_fltHW_bid & inHWMask[0]) && _inCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE1(_regB, _dB, _dBv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            \
            _regB[0] = ( (_fltHW_bid & inHWMask[0]) && _inCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE2(_regB, _dB, _dBv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            _dBv4_off[1] += cInLut.idx[cLut_id]; \
            \
            _regB[0] = ( (_fltHW_bid & inHWMask[0]) && _inCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            _regB[1] = ( (_fltHW_bid & inHWMask[1]) && _inCv16Valid ) ? _dB[ _dBv4_off[1] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE4(_regB, _dB, _dBv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            _dBv4_off[1] += cInLut.idx[cLut_id]; \
            _dBv4_off[2] += cInLut.idx[cLut_id]; \
            _dBv4_off[3] += cInLut.idx[cLut_id]; \
            \
            _regB[0] = ( (_fltHW_bid & inHWMask[0]) && _inCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            _regB[1] = ( (_fltHW_bid & inHWMask[1]) && _inCv16Valid ) ? _dB[ _dBv4_off[1] ] : ZEROv4;\
            _regB[2] = ( (_fltHW_bid & inHWMask[2]) && _inCv16Valid ) ? _dB[ _dBv4_off[2] ] : ZEROv4;\
            _regB[3] = ( (_fltHW_bid & inHWMask[3]) && _inCv16Valid ) ? _dB[ _dBv4_off[3] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE8(_regB, _dB, _dBv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            _dBv4_off[1] += cInLut.idx[cLut_id]; \
            _dBv4_off[2] += cInLut.idx[cLut_id]; \
            _dBv4_off[3] += cInLut.idx[cLut_id]; \
            _dBv4_off[4] += cInLut.idx[cLut_id]; \
            _dBv4_off[5] += cInLut.idx[cLut_id]; \
            _dBv4_off[6] += cInLut.idx[cLut_id]; \
            _dBv4_off[7] += cInLut.idx[cLut_id]; \
            \
            _regB[0] = ( (_fltHW_bid & inHWMask[0]) && _inCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            _regB[1] = ( (_fltHW_bid & inHWMask[1]) && _inCv16Valid ) ? _dB[ _dBv4_off[1] ] : ZEROv4;\
            _regB[2] = ( (_fltHW_bid & inHWMask[2]) && _inCv16Valid ) ? _dB[ _dBv4_off[2] ] : ZEROv4;\
            _regB[3] = ( (_fltHW_bid & inHWMask[3]) && _inCv16Valid ) ? _dB[ _dBv4_off[3] ] : ZEROv4;\
            _regB[4] = ( (_fltHW_bid & inHWMask[4]) && _inCv16Valid ) ? _dB[ _dBv4_off[4] ] : ZEROv4;\
            _regB[5] = ( (_fltHW_bid & inHWMask[5]) && _inCv16Valid ) ? _dB[ _dBv4_off[5] ] : ZEROv4;\
            _regB[6] = ( (_fltHW_bid & inHWMask[6]) && _inCv16Valid ) ? _dB[ _dBv4_off[6] ] : ZEROv4;\
            _regB[7] = ( (_fltHW_bid & inHWMask[7]) && _inCv16Valid ) ? _dB[ _dBv4_off[7] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE16(_regB, _dB, _dBv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dBv4_off[0]  += cInLut.idx[cLut_id]; \
            _dBv4_off[1]  += cInLut.idx[cLut_id]; \
            _dBv4_off[2]  += cInLut.idx[cLut_id]; \
            _dBv4_off[3]  += cInLut.idx[cLut_id]; \
            _dBv4_off[4]  += cInLut.idx[cLut_id]; \
            _dBv4_off[5]  += cInLut.idx[cLut_id]; \
            _dBv4_off[6]  += cInLut.idx[cLut_id]; \
            _dBv4_off[7]  += cInLut.idx[cLut_id]; \
            _dBv4_off[8]  += cInLut.idx[cLut_id]; \
            _dBv4_off[9]  += cInLut.idx[cLut_id]; \
            _dBv4_off[10] += cInLut.idx[cLut_id]; \
            _dBv4_off[11] += cInLut.idx[cLut_id]; \
            _dBv4_off[12] += cInLut.idx[cLut_id]; \
            _dBv4_off[13] += cInLut.idx[cLut_id]; \
            _dBv4_off[14] += cInLut.idx[cLut_id]; \
            _dBv4_off[15] += cInLut.idx[cLut_id]; \
            \
            _regB[0]  = ( (_fltHW_bid & inHWMask[0])  && _inCv16Valid ) ? _dB[ _dBv4_off[0]  ] : ZEROv4;\
            _regB[1]  = ( (_fltHW_bid & inHWMask[1])  && _inCv16Valid ) ? _dB[ _dBv4_off[1]  ] : ZEROv4;\
            _regB[2]  = ( (_fltHW_bid & inHWMask[2])  && _inCv16Valid ) ? _dB[ _dBv4_off[2]  ] : ZEROv4;\
            _regB[3]  = ( (_fltHW_bid & inHWMask[3])  && _inCv16Valid ) ? _dB[ _dBv4_off[3]  ] : ZEROv4;\
            _regB[4]  = ( (_fltHW_bid & inHWMask[4])  && _inCv16Valid ) ? _dB[ _dBv4_off[4]  ] : ZEROv4;\
            _regB[5]  = ( (_fltHW_bid & inHWMask[5])  && _inCv16Valid ) ? _dB[ _dBv4_off[5]  ] : ZEROv4;\
            _regB[6]  = ( (_fltHW_bid & inHWMask[6])  && _inCv16Valid ) ? _dB[ _dBv4_off[6]  ] : ZEROv4;\
            _regB[7]  = ( (_fltHW_bid & inHWMask[7])  && _inCv16Valid ) ? _dB[ _dBv4_off[7]  ] : ZEROv4;\
            _regB[8]  = ( (_fltHW_bid & inHWMask[8])  && _inCv16Valid ) ? _dB[ _dBv4_off[8]  ] : ZEROv4;\
            _regB[9]  = ( (_fltHW_bid & inHWMask[9])  && _inCv16Valid ) ? _dB[ _dBv4_off[9]  ] : ZEROv4;\
            _regB[10] = ( (_fltHW_bid & inHWMask[10]) && _inCv16Valid ) ? _dB[ _dBv4_off[10] ] : ZEROv4;\
            _regB[11] = ( (_fltHW_bid & inHWMask[11]) && _inCv16Valid ) ? _dB[ _dBv4_off[11] ] : ZEROv4;\
            _regB[12] = ( (_fltHW_bid & inHWMask[12]) && _inCv16Valid ) ? _dB[ _dBv4_off[12] ] : ZEROv4;\
            _regB[13] = ( (_fltHW_bid & inHWMask[13]) && _inCv16Valid ) ? _dB[ _dBv4_off[13] ] : ZEROv4;\
            _regB[14] = ( (_fltHW_bid & inHWMask[14]) && _inCv16Valid ) ? _dB[ _dBv4_off[14] ] : ZEROv4;\
            _regB[15] = ( (_fltHW_bid & inHWMask[15]) && _inCv16Valid ) ? _dB[ _dBv4_off[15] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE32(_regB, _dB, _dBv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dBv4_off[0]  += cInLut.idx[cLut_id]; \
            _dBv4_off[1]  += cInLut.idx[cLut_id]; \
            _dBv4_off[2]  += cInLut.idx[cLut_id]; \
            _dBv4_off[3]  += cInLut.idx[cLut_id]; \
            _dBv4_off[4]  += cInLut.idx[cLut_id]; \
            _dBv4_off[5]  += cInLut.idx[cLut_id]; \
            _dBv4_off[6]  += cInLut.idx[cLut_id]; \
            _dBv4_off[7]  += cInLut.idx[cLut_id]; \
            _dBv4_off[8]  += cInLut.idx[cLut_id]; \
            _dBv4_off[9]  += cInLut.idx[cLut_id]; \
            _dBv4_off[10] += cInLut.idx[cLut_id]; \
            _dBv4_off[11] += cInLut.idx[cLut_id]; \
            _dBv4_off[12] += cInLut.idx[cLut_id]; \
            _dBv4_off[13] += cInLut.idx[cLut_id]; \
            _dBv4_off[14] += cInLut.idx[cLut_id]; \
            _dBv4_off[15] += cInLut.idx[cLut_id]; \
            _dBv4_off[16] += cInLut.idx[cLut_id]; \
            _dBv4_off[17] += cInLut.idx[cLut_id]; \
            _dBv4_off[18] += cInLut.idx[cLut_id]; \
            _dBv4_off[19] += cInLut.idx[cLut_id]; \
            _dBv4_off[20] += cInLut.idx[cLut_id]; \
            _dBv4_off[21] += cInLut.idx[cLut_id]; \
            _dBv4_off[22] += cInLut.idx[cLut_id]; \
            _dBv4_off[23] += cInLut.idx[cLut_id]; \
            _dBv4_off[24] += cInLut.idx[cLut_id]; \
            _dBv4_off[25] += cInLut.idx[cLut_id]; \
            _dBv4_off[26] += cInLut.idx[cLut_id]; \
            _dBv4_off[27] += cInLut.idx[cLut_id]; \
            _dBv4_off[28] += cInLut.idx[cLut_id]; \
            _dBv4_off[29] += cInLut.idx[cLut_id]; \
            _dBv4_off[30] += cInLut.idx[cLut_id]; \
            _dBv4_off[31] += cInLut.idx[cLut_id]; \
            \
            _regB[0]  = ( (_fltHW_bid & inHWMask[0])  && _inCv16Valid ) ? _dB[ _dBv4_off[0]  ] : ZEROv4;\
            _regB[1]  = ( (_fltHW_bid & inHWMask[1])  && _inCv16Valid ) ? _dB[ _dBv4_off[1]  ] : ZEROv4;\
            _regB[2]  = ( (_fltHW_bid & inHWMask[2])  && _inCv16Valid ) ? _dB[ _dBv4_off[2]  ] : ZEROv4;\
            _regB[3]  = ( (_fltHW_bid & inHWMask[3])  && _inCv16Valid ) ? _dB[ _dBv4_off[3]  ] : ZEROv4;\
            _regB[4]  = ( (_fltHW_bid & inHWMask[4])  && _inCv16Valid ) ? _dB[ _dBv4_off[4]  ] : ZEROv4;\
            _regB[5]  = ( (_fltHW_bid & inHWMask[5])  && _inCv16Valid ) ? _dB[ _dBv4_off[5]  ] : ZEROv4;\
            _regB[6]  = ( (_fltHW_bid & inHWMask[6])  && _inCv16Valid ) ? _dB[ _dBv4_off[6]  ] : ZEROv4;\
            _regB[7]  = ( (_fltHW_bid & inHWMask[7])  && _inCv16Valid ) ? _dB[ _dBv4_off[7]  ] : ZEROv4;\
            _regB[8]  = ( (_fltHW_bid & inHWMask[8])  && _inCv16Valid ) ? _dB[ _dBv4_off[8]  ] : ZEROv4;\
            _regB[9]  = ( (_fltHW_bid & inHWMask[9])  && _inCv16Valid ) ? _dB[ _dBv4_off[9]  ] : ZEROv4;\
            _regB[10] = ( (_fltHW_bid & inHWMask[10]) && _inCv16Valid ) ? _dB[ _dBv4_off[10] ] : ZEROv4;\
            _regB[11] = ( (_fltHW_bid & inHWMask[11]) && _inCv16Valid ) ? _dB[ _dBv4_off[11] ] : ZEROv4;\
            _regB[12] = ( (_fltHW_bid & inHWMask[12]) && _inCv16Valid ) ? _dB[ _dBv4_off[12] ] : ZEROv4;\
            _regB[13] = ( (_fltHW_bid & inHWMask[13]) && _inCv16Valid ) ? _dB[ _dBv4_off[13] ] : ZEROv4;\
            _regB[14] = ( (_fltHW_bid & inHWMask[14]) && _inCv16Valid ) ? _dB[ _dBv4_off[14] ] : ZEROv4;\
            _regB[15] = ( (_fltHW_bid & inHWMask[15]) && _inCv16Valid ) ? _dB[ _dBv4_off[15] ] : ZEROv4;\
            _regB[16] = ( (_fltHW_bid & inHWMask[16]) && _inCv16Valid ) ? _dB[ _dBv4_off[16] ] : ZEROv4;\
            _regB[17] = ( (_fltHW_bid & inHWMask[17]) && _inCv16Valid ) ? _dB[ _dBv4_off[17] ] : ZEROv4;\
            _regB[18] = ( (_fltHW_bid & inHWMask[18]) && _inCv16Valid ) ? _dB[ _dBv4_off[18] ] : ZEROv4;\
            _regB[19] = ( (_fltHW_bid & inHWMask[19]) && _inCv16Valid ) ? _dB[ _dBv4_off[19] ] : ZEROv4;\
            _regB[20] = ( (_fltHW_bid & inHWMask[20]) && _inCv16Valid ) ? _dB[ _dBv4_off[20] ] : ZEROv4;\
            _regB[21] = ( (_fltHW_bid & inHWMask[21]) && _inCv16Valid ) ? _dB[ _dBv4_off[21] ] : ZEROv4;\
            _regB[22] = ( (_fltHW_bid & inHWMask[22]) && _inCv16Valid ) ? _dB[ _dBv4_off[22] ] : ZEROv4;\
            _regB[23] = ( (_fltHW_bid & inHWMask[23]) && _inCv16Valid ) ? _dB[ _dBv4_off[23] ] : ZEROv4;\
            _regB[24] = ( (_fltHW_bid & inHWMask[24]) && _inCv16Valid ) ? _dB[ _dBv4_off[24] ] : ZEROv4;\
            _regB[25] = ( (_fltHW_bid & inHWMask[25]) && _inCv16Valid ) ? _dB[ _dBv4_off[25] ] : ZEROv4;\
            _regB[26] = ( (_fltHW_bid & inHWMask[26]) && _inCv16Valid ) ? _dB[ _dBv4_off[26] ] : ZEROv4;\
            _regB[27] = ( (_fltHW_bid & inHWMask[27]) && _inCv16Valid ) ? _dB[ _dBv4_off[27] ] : ZEROv4;\
            _regB[28] = ( (_fltHW_bid & inHWMask[28]) && _inCv16Valid ) ? _dB[ _dBv4_off[28] ] : ZEROv4;\
            _regB[29] = ( (_fltHW_bid & inHWMask[29]) && _inCv16Valid ) ? _dB[ _dBv4_off[29] ] : ZEROv4;\
            _regB[30] = ( (_fltHW_bid & inHWMask[30]) && _inCv16Valid ) ? _dB[ _dBv4_off[30] ] : ZEROv4;\
            _regB[31] = ( (_fltHW_bid & inHWMask[31]) && _inCv16Valid ) ? _dB[ _dBv4_off[31] ] : ZEROv4;\
        }

