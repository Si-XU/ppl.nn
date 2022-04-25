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
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE_8TH(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE_QTR(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE_HALF(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE1(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE2(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            _dBv4_off[1] += cFltLut.idx[cLut_id]; \
            \
            _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            _regB[1] = ( _fltNValid[1] && _fltCv16Valid ) ? _dB[ _dBv4_off[1] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE4(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            _dBv4_off[1] += cFltLut.idx[cLut_id]; \
            _dBv4_off[2] += cFltLut.idx[cLut_id]; \
            _dBv4_off[3] += cFltLut.idx[cLut_id]; \
            \
            _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            _regB[1] = ( _fltNValid[1] && _fltCv16Valid ) ? _dB[ _dBv4_off[1] ] : ZEROv4;\
            _regB[2] = ( _fltNValid[2] && _fltCv16Valid ) ? _dB[ _dBv4_off[2] ] : ZEROv4;\
            _regB[3] = ( _fltNValid[3] && _fltCv16Valid ) ? _dB[ _dBv4_off[3] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE8(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            _dBv4_off[1] += cFltLut.idx[cLut_id]; \
            _dBv4_off[2] += cFltLut.idx[cLut_id]; \
            _dBv4_off[3] += cFltLut.idx[cLut_id]; \
            _dBv4_off[4] += cFltLut.idx[cLut_id]; \
            _dBv4_off[5] += cFltLut.idx[cLut_id]; \
            _dBv4_off[6] += cFltLut.idx[cLut_id]; \
            _dBv4_off[7] += cFltLut.idx[cLut_id]; \
            \
            _regB[0] = ( _fltNValid[0] && _fltCv16Valid ) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            _regB[1] = ( _fltNValid[1] && _fltCv16Valid ) ? _dB[ _dBv4_off[1] ] : ZEROv4;\
            _regB[2] = ( _fltNValid[2] && _fltCv16Valid ) ? _dB[ _dBv4_off[2] ] : ZEROv4;\
            _regB[3] = ( _fltNValid[3] && _fltCv16Valid ) ? _dB[ _dBv4_off[3] ] : ZEROv4;\
            _regB[4] = ( _fltNValid[4] && _fltCv16Valid ) ? _dB[ _dBv4_off[4] ] : ZEROv4;\
            _regB[5] = ( _fltNValid[5] && _fltCv16Valid ) ? _dB[ _dBv4_off[5] ] : ZEROv4;\
            _regB[6] = ( _fltNValid[6] && _fltCv16Valid ) ? _dB[ _dBv4_off[6] ] : ZEROv4;\
            _regB[7] = ( _fltNValid[7] && _fltCv16Valid ) ? _dB[ _dBv4_off[7] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE16(_regB, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0]  += cFltLut.idx[cLut_id]; \
            _dBv4_off[1]  += cFltLut.idx[cLut_id]; \
            _dBv4_off[2]  += cFltLut.idx[cLut_id]; \
            _dBv4_off[3]  += cFltLut.idx[cLut_id]; \
            _dBv4_off[4]  += cFltLut.idx[cLut_id]; \
            _dBv4_off[5]  += cFltLut.idx[cLut_id]; \
            _dBv4_off[6]  += cFltLut.idx[cLut_id]; \
            _dBv4_off[7]  += cFltLut.idx[cLut_id]; \
            _dBv4_off[8]  += cFltLut.idx[cLut_id]; \
            _dBv4_off[9]  += cFltLut.idx[cLut_id]; \
            _dBv4_off[10] += cFltLut.idx[cLut_id]; \
            _dBv4_off[11] += cFltLut.idx[cLut_id]; \
            _dBv4_off[12] += cFltLut.idx[cLut_id]; \
            _dBv4_off[13] += cFltLut.idx[cLut_id]; \
            _dBv4_off[14] += cFltLut.idx[cLut_id]; \
            _dBv4_off[15] += cFltLut.idx[cLut_id]; \
            \
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
        }

#define SET_dBv4_BOUND(_step_id, _dBv4_off, _fltNValid) \
        { \
            int _fltN_id  =  cta_idx  *  TILE_N_PER_CTA + \
                            _step_id  * (TILE_N_PER_CTA / READ_dBv4_STEPS) + \
                             ldg_idy; \
            \
            _fltNValid  =  _fltN_id < numFltPerGrpPad; \
            \
            _dBv4_off  =   grp_id   * fltHW * numChlPerGrpPadV16 * numFltPerGrpPad + \
                          _fltN_id  * fltHW * numChlPerGrpPadV16 + \
                           fltCv16_id; \
        }

////////////////////////////////////////
// load dA macros
////////////////////////////////////////

#define SET_dAv4_BOUND(_step_id, _dAv4_off, _inN_id, _inH_START, _inW_START) \
        { \
            int _outNHW_id    =  cta_idy  *  TILE_M_PER_CTA + \
                                _step_id  * (TILE_M_PER_CTA / READ_dAv4_STEPS) + \
                                 ldg_idy; \
            \
            int _outW_id =  (_outNHW_id % outWidth); \
            int _outH_id =  (_outNHW_id / outWidth) % outHeight; \
            int _inH_id  =     _outH_id * strideHeight; \
            int _inW_id  =     _outW_id * strideWidth; \
            \
            _inN_id      =  _outNHW_id / outHW; \
            _inH_START   =  _inH_id - padHeight; \
            _inW_START   =  _inW_id - padWidth;  \
            \
            _dAv4_off  =  (_inN_id  * inHW + _inH_id  * inWidth + _inW_id) * numChlPerGrpPadV16 * numGrp + \
                           grp_id   * numChlPerGrpPadV16 + \
                           fltCv16_id; \
        }

#define LOAD_dAv4_SIZE_16TH(_regA, _dA, _dAv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                _regA[0] = ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE_8TH(_regA, _dA, _dAv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                _regA[0] = ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE_QTR(_regA, _dA, _dAv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                _regA[0] = ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE_HALF(_regA, _dA, _dAv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                _regA[0] = ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE1(_regA, _dA, _dAv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            \
            _regA[0] = ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE2(_regA, _dA, _dAv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            _dAv4_off[1] += cInLut.idx[cLut_id]; \
            \
            _regA[0] = ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            _regA[1] = ( fltCv16Valid && HeightInRange(_inH_id[1]) && WidthInRange(_inW_id[1]) && (_inN_id[1] < inNum) ) ? _dA[ _dAv4_off[1] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE4(_regA, _dA, _dAv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            _dAv4_off[1] += cInLut.idx[cLut_id]; \
            _dAv4_off[2] += cInLut.idx[cLut_id]; \
            _dAv4_off[3] += cInLut.idx[cLut_id]; \
            \
            _regA[0] = ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            _regA[1] = ( fltCv16Valid && HeightInRange(_inH_id[1]) && WidthInRange(_inW_id[1]) && (_inN_id[1] < inNum) ) ? _dA[ _dAv4_off[1] ] : ZEROv4;\
            _regA[2] = ( fltCv16Valid && HeightInRange(_inH_id[2]) && WidthInRange(_inW_id[2]) && (_inN_id[2] < inNum) ) ? _dA[ _dAv4_off[2] ] : ZEROv4;\
            _regA[3] = ( fltCv16Valid && HeightInRange(_inH_id[3]) && WidthInRange(_inW_id[3]) && (_inN_id[3] < inNum) ) ? _dA[ _dAv4_off[3] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE8(_regA, _dA, _dAv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            _dAv4_off[1] += cInLut.idx[cLut_id]; \
            _dAv4_off[2] += cInLut.idx[cLut_id]; \
            _dAv4_off[3] += cInLut.idx[cLut_id]; \
            _dAv4_off[4] += cInLut.idx[cLut_id]; \
            _dAv4_off[5] += cInLut.idx[cLut_id]; \
            _dAv4_off[6] += cInLut.idx[cLut_id]; \
            _dAv4_off[7] += cInLut.idx[cLut_id]; \
            \
            _regA[0] = ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            _regA[1] = ( fltCv16Valid && HeightInRange(_inH_id[1]) && WidthInRange(_inW_id[1]) && (_inN_id[1] < inNum) ) ? _dA[ _dAv4_off[1] ] : ZEROv4;\
            _regA[2] = ( fltCv16Valid && HeightInRange(_inH_id[2]) && WidthInRange(_inW_id[2]) && (_inN_id[2] < inNum) ) ? _dA[ _dAv4_off[2] ] : ZEROv4;\
            _regA[3] = ( fltCv16Valid && HeightInRange(_inH_id[3]) && WidthInRange(_inW_id[3]) && (_inN_id[3] < inNum) ) ? _dA[ _dAv4_off[3] ] : ZEROv4;\
            _regA[4] = ( fltCv16Valid && HeightInRange(_inH_id[4]) && WidthInRange(_inW_id[4]) && (_inN_id[4] < inNum) ) ? _dA[ _dAv4_off[4] ] : ZEROv4;\
            _regA[5] = ( fltCv16Valid && HeightInRange(_inH_id[5]) && WidthInRange(_inW_id[5]) && (_inN_id[5] < inNum) ) ? _dA[ _dAv4_off[5] ] : ZEROv4;\
            _regA[6] = ( fltCv16Valid && HeightInRange(_inH_id[6]) && WidthInRange(_inW_id[6]) && (_inN_id[6] < inNum) ) ? _dA[ _dAv4_off[6] ] : ZEROv4;\
            _regA[7] = ( fltCv16Valid && HeightInRange(_inH_id[7]) && WidthInRange(_inW_id[7]) && (_inN_id[7] < inNum) ) ? _dA[ _dAv4_off[7] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE16(_regA, _dA, _dAv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dAv4_off[0]  += cInLut.idx[cLut_id]; \
            _dAv4_off[1]  += cInLut.idx[cLut_id]; \
            _dAv4_off[2]  += cInLut.idx[cLut_id]; \
            _dAv4_off[3]  += cInLut.idx[cLut_id]; \
            _dAv4_off[4]  += cInLut.idx[cLut_id]; \
            _dAv4_off[5]  += cInLut.idx[cLut_id]; \
            _dAv4_off[6]  += cInLut.idx[cLut_id]; \
            _dAv4_off[7]  += cInLut.idx[cLut_id]; \
            _dAv4_off[8]  += cInLut.idx[cLut_id]; \
            _dAv4_off[9]  += cInLut.idx[cLut_id]; \
            _dAv4_off[10] += cInLut.idx[cLut_id]; \
            _dAv4_off[11] += cInLut.idx[cLut_id]; \
            _dAv4_off[12] += cInLut.idx[cLut_id]; \
            _dAv4_off[13] += cInLut.idx[cLut_id]; \
            _dAv4_off[14] += cInLut.idx[cLut_id]; \
            _dAv4_off[15] += cInLut.idx[cLut_id]; \
            \
            _regA[0]  = ( fltCv16Valid && HeightInRange(_inH_id[0])  && WidthInRange(_inW_id[0])  && (_inN_id[0]  < inNum) ) ? _dA[ _dAv4_off[0] ]  : ZEROv4;\
            _regA[1]  = ( fltCv16Valid && HeightInRange(_inH_id[1])  && WidthInRange(_inW_id[1])  && (_inN_id[1]  < inNum) ) ? _dA[ _dAv4_off[1] ]  : ZEROv4;\
            _regA[2]  = ( fltCv16Valid && HeightInRange(_inH_id[2])  && WidthInRange(_inW_id[2])  && (_inN_id[2]  < inNum) ) ? _dA[ _dAv4_off[2] ]  : ZEROv4;\
            _regA[3]  = ( fltCv16Valid && HeightInRange(_inH_id[3])  && WidthInRange(_inW_id[3])  && (_inN_id[3]  < inNum) ) ? _dA[ _dAv4_off[3] ]  : ZEROv4;\
            _regA[4]  = ( fltCv16Valid && HeightInRange(_inH_id[4])  && WidthInRange(_inW_id[4])  && (_inN_id[4]  < inNum) ) ? _dA[ _dAv4_off[4] ]  : ZEROv4;\
            _regA[5]  = ( fltCv16Valid && HeightInRange(_inH_id[5])  && WidthInRange(_inW_id[5])  && (_inN_id[5]  < inNum) ) ? _dA[ _dAv4_off[5] ]  : ZEROv4;\
            _regA[6]  = ( fltCv16Valid && HeightInRange(_inH_id[6])  && WidthInRange(_inW_id[6])  && (_inN_id[6]  < inNum) ) ? _dA[ _dAv4_off[6] ]  : ZEROv4;\
            _regA[7]  = ( fltCv16Valid && HeightInRange(_inH_id[7])  && WidthInRange(_inW_id[7])  && (_inN_id[7]  < inNum) ) ? _dA[ _dAv4_off[7] ]  : ZEROv4;\
            _regA[8]  = ( fltCv16Valid && HeightInRange(_inH_id[8])  && WidthInRange(_inW_id[8])  && (_inN_id[8]  < inNum) ) ? _dA[ _dAv4_off[8] ]  : ZEROv4;\
            _regA[9]  = ( fltCv16Valid && HeightInRange(_inH_id[9])  && WidthInRange(_inW_id[9])  && (_inN_id[9]  < inNum) ) ? _dA[ _dAv4_off[9] ]  : ZEROv4;\
            _regA[10] = ( fltCv16Valid && HeightInRange(_inH_id[10]) && WidthInRange(_inW_id[10]) && (_inN_id[10] < inNum) ) ? _dA[ _dAv4_off[10] ] : ZEROv4;\
            _regA[11] = ( fltCv16Valid && HeightInRange(_inH_id[11]) && WidthInRange(_inW_id[11]) && (_inN_id[11] < inNum) ) ? _dA[ _dAv4_off[11] ] : ZEROv4;\
            _regA[12] = ( fltCv16Valid && HeightInRange(_inH_id[12]) && WidthInRange(_inW_id[12]) && (_inN_id[12] < inNum) ) ? _dA[ _dAv4_off[12] ] : ZEROv4;\
            _regA[13] = ( fltCv16Valid && HeightInRange(_inH_id[13]) && WidthInRange(_inW_id[13]) && (_inN_id[13] < inNum) ) ? _dA[ _dAv4_off[13] ] : ZEROv4;\
            _regA[14] = ( fltCv16Valid && HeightInRange(_inH_id[14]) && WidthInRange(_inW_id[14]) && (_inN_id[14] < inNum) ) ? _dA[ _dAv4_off[14] ] : ZEROv4;\
            _regA[15] = ( fltCv16Valid && HeightInRange(_inH_id[15]) && WidthInRange(_inW_id[15]) && (_inN_id[15] < inNum) ) ? _dA[ _dAv4_off[15] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE32(_regA, _dA, _dAv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dAv4_off[0]  += cInLut.idx[cLut_id]; \
            _dAv4_off[1]  += cInLut.idx[cLut_id]; \
            _dAv4_off[2]  += cInLut.idx[cLut_id]; \
            _dAv4_off[3]  += cInLut.idx[cLut_id]; \
            _dAv4_off[4]  += cInLut.idx[cLut_id]; \
            _dAv4_off[5]  += cInLut.idx[cLut_id]; \
            _dAv4_off[6]  += cInLut.idx[cLut_id]; \
            _dAv4_off[7]  += cInLut.idx[cLut_id]; \
            _dAv4_off[8]  += cInLut.idx[cLut_id]; \
            _dAv4_off[9]  += cInLut.idx[cLut_id]; \
            _dAv4_off[10] += cInLut.idx[cLut_id]; \
            _dAv4_off[11] += cInLut.idx[cLut_id]; \
            _dAv4_off[12] += cInLut.idx[cLut_id]; \
            _dAv4_off[13] += cInLut.idx[cLut_id]; \
            _dAv4_off[14] += cInLut.idx[cLut_id]; \
            _dAv4_off[15] += cInLut.idx[cLut_id]; \
            _dAv4_off[16] += cInLut.idx[cLut_id]; \
            _dAv4_off[17] += cInLut.idx[cLut_id]; \
            _dAv4_off[18] += cInLut.idx[cLut_id]; \
            _dAv4_off[19] += cInLut.idx[cLut_id]; \
            _dAv4_off[20] += cInLut.idx[cLut_id]; \
            _dAv4_off[21] += cInLut.idx[cLut_id]; \
            _dAv4_off[22] += cInLut.idx[cLut_id]; \
            _dAv4_off[23] += cInLut.idx[cLut_id]; \
            _dAv4_off[24] += cInLut.idx[cLut_id]; \
            _dAv4_off[25] += cInLut.idx[cLut_id]; \
            _dAv4_off[26] += cInLut.idx[cLut_id]; \
            _dAv4_off[27] += cInLut.idx[cLut_id]; \
            _dAv4_off[28] += cInLut.idx[cLut_id]; \
            _dAv4_off[29] += cInLut.idx[cLut_id]; \
            _dAv4_off[30] += cInLut.idx[cLut_id]; \
            _dAv4_off[31] += cInLut.idx[cLut_id]; \
            \
            _regA[0]  = ( fltCv16Valid && HeightInRange(_inH_id[0])  && WidthInRange(_inW_id[0])  && (_inN_id[0]  < inNum) ) ? _dA[ _dAv4_off[0] ]  : ZEROv4;\
            _regA[1]  = ( fltCv16Valid && HeightInRange(_inH_id[1])  && WidthInRange(_inW_id[1])  && (_inN_id[1]  < inNum) ) ? _dA[ _dAv4_off[1] ]  : ZEROv4;\
            _regA[2]  = ( fltCv16Valid && HeightInRange(_inH_id[2])  && WidthInRange(_inW_id[2])  && (_inN_id[2]  < inNum) ) ? _dA[ _dAv4_off[2] ]  : ZEROv4;\
            _regA[3]  = ( fltCv16Valid && HeightInRange(_inH_id[3])  && WidthInRange(_inW_id[3])  && (_inN_id[3]  < inNum) ) ? _dA[ _dAv4_off[3] ]  : ZEROv4;\
            _regA[4]  = ( fltCv16Valid && HeightInRange(_inH_id[4])  && WidthInRange(_inW_id[4])  && (_inN_id[4]  < inNum) ) ? _dA[ _dAv4_off[4] ]  : ZEROv4;\
            _regA[5]  = ( fltCv16Valid && HeightInRange(_inH_id[5])  && WidthInRange(_inW_id[5])  && (_inN_id[5]  < inNum) ) ? _dA[ _dAv4_off[5] ]  : ZEROv4;\
            _regA[6]  = ( fltCv16Valid && HeightInRange(_inH_id[6])  && WidthInRange(_inW_id[6])  && (_inN_id[6]  < inNum) ) ? _dA[ _dAv4_off[6] ]  : ZEROv4;\
            _regA[7]  = ( fltCv16Valid && HeightInRange(_inH_id[7])  && WidthInRange(_inW_id[7])  && (_inN_id[7]  < inNum) ) ? _dA[ _dAv4_off[7] ]  : ZEROv4;\
            _regA[8]  = ( fltCv16Valid && HeightInRange(_inH_id[8])  && WidthInRange(_inW_id[8])  && (_inN_id[8]  < inNum) ) ? _dA[ _dAv4_off[8] ]  : ZEROv4;\
            _regA[9]  = ( fltCv16Valid && HeightInRange(_inH_id[9])  && WidthInRange(_inW_id[9])  && (_inN_id[9]  < inNum) ) ? _dA[ _dAv4_off[9] ]  : ZEROv4;\
            _regA[10] = ( fltCv16Valid && HeightInRange(_inH_id[10]) && WidthInRange(_inW_id[10]) && (_inN_id[10] < inNum) ) ? _dA[ _dAv4_off[10] ] : ZEROv4;\
            _regA[11] = ( fltCv16Valid && HeightInRange(_inH_id[11]) && WidthInRange(_inW_id[11]) && (_inN_id[11] < inNum) ) ? _dA[ _dAv4_off[11] ] : ZEROv4;\
            _regA[12] = ( fltCv16Valid && HeightInRange(_inH_id[12]) && WidthInRange(_inW_id[12]) && (_inN_id[12] < inNum) ) ? _dA[ _dAv4_off[12] ] : ZEROv4;\
            _regA[13] = ( fltCv16Valid && HeightInRange(_inH_id[13]) && WidthInRange(_inW_id[13]) && (_inN_id[13] < inNum) ) ? _dA[ _dAv4_off[13] ] : ZEROv4;\
            _regA[14] = ( fltCv16Valid && HeightInRange(_inH_id[14]) && WidthInRange(_inW_id[14]) && (_inN_id[14] < inNum) ) ? _dA[ _dAv4_off[14] ] : ZEROv4;\
            _regA[15] = ( fltCv16Valid && HeightInRange(_inH_id[15]) && WidthInRange(_inW_id[15]) && (_inN_id[15] < inNum) ) ? _dA[ _dAv4_off[15] ] : ZEROv4;\
            _regA[16] = ( fltCv16Valid && HeightInRange(_inH_id[16]) && WidthInRange(_inW_id[16]) && (_inN_id[16] < inNum) ) ? _dA[ _dAv4_off[16] ] : ZEROv4;\
            _regA[17] = ( fltCv16Valid && HeightInRange(_inH_id[17]) && WidthInRange(_inW_id[17]) && (_inN_id[17] < inNum) ) ? _dA[ _dAv4_off[17] ] : ZEROv4;\
            _regA[18] = ( fltCv16Valid && HeightInRange(_inH_id[18]) && WidthInRange(_inW_id[18]) && (_inN_id[18] < inNum) ) ? _dA[ _dAv4_off[18] ] : ZEROv4;\
            _regA[19] = ( fltCv16Valid && HeightInRange(_inH_id[19]) && WidthInRange(_inW_id[19]) && (_inN_id[19] < inNum) ) ? _dA[ _dAv4_off[19] ] : ZEROv4;\
            _regA[20] = ( fltCv16Valid && HeightInRange(_inH_id[20]) && WidthInRange(_inW_id[20]) && (_inN_id[20] < inNum) ) ? _dA[ _dAv4_off[20] ] : ZEROv4;\
            _regA[21] = ( fltCv16Valid && HeightInRange(_inH_id[21]) && WidthInRange(_inW_id[21]) && (_inN_id[21] < inNum) ) ? _dA[ _dAv4_off[21] ] : ZEROv4;\
            _regA[22] = ( fltCv16Valid && HeightInRange(_inH_id[22]) && WidthInRange(_inW_id[22]) && (_inN_id[22] < inNum) ) ? _dA[ _dAv4_off[22] ] : ZEROv4;\
            _regA[23] = ( fltCv16Valid && HeightInRange(_inH_id[23]) && WidthInRange(_inW_id[23]) && (_inN_id[23] < inNum) ) ? _dA[ _dAv4_off[23] ] : ZEROv4;\
            _regA[24] = ( fltCv16Valid && HeightInRange(_inH_id[24]) && WidthInRange(_inW_id[24]) && (_inN_id[24] < inNum) ) ? _dA[ _dAv4_off[24] ] : ZEROv4;\
            _regA[25] = ( fltCv16Valid && HeightInRange(_inH_id[25]) && WidthInRange(_inW_id[25]) && (_inN_id[25] < inNum) ) ? _dA[ _dAv4_off[25] ] : ZEROv4;\
            _regA[26] = ( fltCv16Valid && HeightInRange(_inH_id[26]) && WidthInRange(_inW_id[26]) && (_inN_id[26] < inNum) ) ? _dA[ _dAv4_off[26] ] : ZEROv4;\
            _regA[27] = ( fltCv16Valid && HeightInRange(_inH_id[27]) && WidthInRange(_inW_id[27]) && (_inN_id[27] < inNum) ) ? _dA[ _dAv4_off[27] ] : ZEROv4;\
            _regA[28] = ( fltCv16Valid && HeightInRange(_inH_id[28]) && WidthInRange(_inW_id[28]) && (_inN_id[28] < inNum) ) ? _dA[ _dAv4_off[28] ] : ZEROv4;\
            _regA[29] = ( fltCv16Valid && HeightInRange(_inH_id[29]) && WidthInRange(_inW_id[29]) && (_inN_id[29] < inNum) ) ? _dA[ _dAv4_off[29] ] : ZEROv4;\
            _regA[30] = ( fltCv16Valid && HeightInRange(_inH_id[30]) && WidthInRange(_inW_id[30]) && (_inN_id[30] < inNum) ) ? _dA[ _dAv4_off[30] ] : ZEROv4;\
            _regA[31] = ( fltCv16Valid && HeightInRange(_inH_id[31]) && WidthInRange(_inW_id[31]) && (_inN_id[31] < inNum) ) ? _dA[ _dAv4_off[31] ] : ZEROv4;\
        }

