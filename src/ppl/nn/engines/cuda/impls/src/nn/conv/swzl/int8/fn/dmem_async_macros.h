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

#define LOAD_dAv4_SIZE_16TH(_sAv4, _sAv4_off, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE_8TH(_sAv4, _sAv4_off, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE_QTR(_sAv4, _sAv4_off, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE_HALF(_sAv4, _sAv4_off, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE1(_sAv4, _sAv4_off, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE2(_sAv4, _sAv4_off, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            _dAv4_off[1] += cFltLut.idx[cLut_id]; \
            \
            CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            CP_ASYNC( (_fltNValid[1] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1, _dA, _dAv4_off[1]); \
        }

#define LOAD_dAv4_SIZE4(_sAv4, _sAv4_off, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dAv4_off[0] += cFltLut.idx[cLut_id]; \
            _dAv4_off[1] += cFltLut.idx[cLut_id]; \
            _dAv4_off[2] += cFltLut.idx[cLut_id]; \
            _dAv4_off[3] += cFltLut.idx[cLut_id]; \
            \
            CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            CP_ASYNC( (_fltNValid[1] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1, _dA, _dAv4_off[1]); \
            CP_ASYNC( (_fltNValid[2] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2, _dA, _dAv4_off[2]); \
            CP_ASYNC( (_fltNValid[3] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3, _dA, _dAv4_off[3]); \
        }

#define LOAD_dAv4_SIZE8(_sAv4, _sAv4_off, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
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
            CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            CP_ASYNC( (_fltNValid[1] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1, _dA, _dAv4_off[1]); \
            CP_ASYNC( (_fltNValid[2] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2, _dA, _dAv4_off[2]); \
            CP_ASYNC( (_fltNValid[3] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3, _dA, _dAv4_off[3]); \
            CP_ASYNC( (_fltNValid[4] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 4, _dA, _dAv4_off[4]); \
            CP_ASYNC( (_fltNValid[5] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 5, _dA, _dAv4_off[5]); \
            CP_ASYNC( (_fltNValid[6] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 6, _dA, _dAv4_off[6]); \
            CP_ASYNC( (_fltNValid[7] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 7, _dA, _dAv4_off[7]); \
        }

#define LOAD_dAv4_SIZE16(_sAv4, _sAv4_off, _dA, _dAv4_off, _fltCv16Valid, _fltNValid) \
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
            CP_ASYNC( (_fltNValid[0]  && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0,  _dA, _dAv4_off[0]);  \
            CP_ASYNC( (_fltNValid[1]  && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1,  _dA, _dAv4_off[1]);  \
            CP_ASYNC( (_fltNValid[2]  && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2,  _dA, _dAv4_off[2]);  \
            CP_ASYNC( (_fltNValid[3]  && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3,  _dA, _dAv4_off[3]);  \
            CP_ASYNC( (_fltNValid[4]  && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 4,  _dA, _dAv4_off[4]);  \
            CP_ASYNC( (_fltNValid[5]  && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 5,  _dA, _dAv4_off[5]);  \
            CP_ASYNC( (_fltNValid[6]  && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 6,  _dA, _dAv4_off[6]);  \
            CP_ASYNC( (_fltNValid[7]  && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 7,  _dA, _dAv4_off[7]);  \
            CP_ASYNC( (_fltNValid[8]  && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 8,  _dA, _dAv4_off[8]);  \
            CP_ASYNC( (_fltNValid[9]  && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 9,  _dA, _dAv4_off[9]);  \
            CP_ASYNC( (_fltNValid[10] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 10, _dA, _dAv4_off[10]); \
            CP_ASYNC( (_fltNValid[11] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 11, _dA, _dAv4_off[11]); \
            CP_ASYNC( (_fltNValid[12] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 12, _dA, _dAv4_off[12]); \
            CP_ASYNC( (_fltNValid[13] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 13, _dA, _dAv4_off[13]); \
            CP_ASYNC( (_fltNValid[14] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 14, _dA, _dAv4_off[14]); \
            CP_ASYNC( (_fltNValid[15] && _fltCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 15, _dA, _dAv4_off[15]); \
        }

#define SET_dAv4_BOUND(_step_id, _dAv4_off, _fltNValid) \
        { \
            int _fltN_id  =  cta_idy  *  TILE_M_PER_CTA + \
                            _step_id  * (TILE_M_PER_CTA / READ_dAv4_STEPS) + \
                             ldg_idy; \
            \
            _fltNValid  =  _fltN_id < numFltPerGrpPad; \
            \
            _dAv4_off  =   grp_id   * fltHW * numChlPerGrpPadV16 * numFltPerGrpPad + \
                          _fltN_id  * fltHW * numChlPerGrpPadV16 + \
                           fltCv16_id; \
        }

////////////////////////////////////////
// load dB macros
////////////////////////////////////////

#define SET_dBv4_BOUND(_step_id, _dBv4_off, _inN_id, _inH_START, _inW_START) \
        { \
            int _outNHW_id    =  cta_idx  *  TILE_N_PER_CTA + \
                                _step_id  * (TILE_N_PER_CTA / READ_dBv4_STEPS) + \
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
            _dBv4_off  =  (_inN_id  * inHW + _inH_id  * inWidth + _inW_id) * numChlPerGrpPadV16 * numGrp + \
                           grp_id   * numChlPerGrpPadV16 + \
                           fltCv16_id; \
        }

#define LOAD_dBv4_SIZE_16TH(_sBv4, _sBv4_off, _dB, _dBv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE_8TH(_sBv4, _sBv4_off, _dB, _dBv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE_QTR(_sBv4, _sBv4_off, _dB, _dBv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE_HALF(_sBv4, _sBv4_off, _dB, _dBv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE1(_sBv4, _sBv4_off, _dB, _dBv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE2(_sBv4, _sBv4_off, _dB, _dBv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            _dBv4_off[1] += cInLut.idx[cLut_id]; \
            \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[1]) && WidthInRange(_inW_id[1]) && (_inN_id[1] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]); \
        }

#define LOAD_dBv4_SIZE4(_sBv4, _sBv4_off, _dB, _dBv4_off, _inN_id, _inH_id, _inW_id) \
        { \
            _dBv4_off[0] += cInLut.idx[cLut_id]; \
            _dBv4_off[1] += cInLut.idx[cLut_id]; \
            _dBv4_off[2] += cInLut.idx[cLut_id]; \
            _dBv4_off[3] += cInLut.idx[cLut_id]; \
            \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[1]) && WidthInRange(_inW_id[1]) && (_inN_id[1] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[2]) && WidthInRange(_inW_id[2]) && (_inN_id[2] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2, _dB, _dBv4_off[2]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[3]) && WidthInRange(_inW_id[3]) && (_inN_id[3] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3, _dB, _dBv4_off[3]); \
        }

#define LOAD_dBv4_SIZE8(_sBv4, _sBv4_off, _dB, _dBv4_off, _inN_id, _inH_id, _inW_id) \
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
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[0]) && WidthInRange(_inW_id[0]) && (_inN_id[0] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[1]) && WidthInRange(_inW_id[1]) && (_inN_id[1] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[2]) && WidthInRange(_inW_id[2]) && (_inN_id[2] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2, _dB, _dBv4_off[2]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[3]) && WidthInRange(_inW_id[3]) && (_inN_id[3] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3, _dB, _dBv4_off[3]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[4]) && WidthInRange(_inW_id[4]) && (_inN_id[4] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 4, _dB, _dBv4_off[4]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[5]) && WidthInRange(_inW_id[5]) && (_inN_id[5] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 5, _dB, _dBv4_off[5]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[6]) && WidthInRange(_inW_id[6]) && (_inN_id[6] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 6, _dB, _dBv4_off[6]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[7]) && WidthInRange(_inW_id[7]) && (_inN_id[7] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 7, _dB, _dBv4_off[7]); \
        }

#define LOAD_dBv4_SIZE16(_sBv4, _sBv4_off, _dB, _dBv4_off, _inN_id, _inH_id, _inW_id) \
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
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[0])  && WidthInRange(_inW_id[0])  && (_inN_id[0]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0,  _dB, _dBv4_off[0]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[1])  && WidthInRange(_inW_id[1])  && (_inN_id[1]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1,  _dB, _dBv4_off[1]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[2])  && WidthInRange(_inW_id[2])  && (_inN_id[2]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2,  _dB, _dBv4_off[2]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[3])  && WidthInRange(_inW_id[3])  && (_inN_id[3]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3,  _dB, _dBv4_off[3]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[4])  && WidthInRange(_inW_id[4])  && (_inN_id[4]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 4,  _dB, _dBv4_off[4]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[5])  && WidthInRange(_inW_id[5])  && (_inN_id[5]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 5,  _dB, _dBv4_off[5]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[6])  && WidthInRange(_inW_id[6])  && (_inN_id[6]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 6,  _dB, _dBv4_off[6]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[7])  && WidthInRange(_inW_id[7])  && (_inN_id[7]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 7,  _dB, _dBv4_off[7]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[8])  && WidthInRange(_inW_id[8])  && (_inN_id[8]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 8,  _dB, _dBv4_off[8]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[9])  && WidthInRange(_inW_id[9])  && (_inN_id[9]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 9,  _dB, _dBv4_off[9]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[10]) && WidthInRange(_inW_id[10]) && (_inN_id[10] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 10, _dB, _dBv4_off[10]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[11]) && WidthInRange(_inW_id[11]) && (_inN_id[11] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 11, _dB, _dBv4_off[11]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[12]) && WidthInRange(_inW_id[12]) && (_inN_id[12] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 12, _dB, _dBv4_off[12]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[13]) && WidthInRange(_inW_id[13]) && (_inN_id[13] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 13, _dB, _dBv4_off[13]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[14]) && WidthInRange(_inW_id[14]) && (_inN_id[14] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 14, _dB, _dBv4_off[14]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[15]) && WidthInRange(_inW_id[15]) && (_inN_id[15] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 15, _dB, _dBv4_off[15]); \
        }

#define LOAD_dBv4_SIZE32(_sBv4, _sBv4_off, _dB, _dBv4_off, _inN_id, _inH_id, _inW_id) \
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
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[0])  && WidthInRange(_inW_id[0])  && (_inN_id[0]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0,  _dB, _dBv4_off[0]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[1])  && WidthInRange(_inW_id[1])  && (_inN_id[1]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1,  _dB, _dBv4_off[1]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[2])  && WidthInRange(_inW_id[2])  && (_inN_id[2]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2,  _dB, _dBv4_off[2]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[3])  && WidthInRange(_inW_id[3])  && (_inN_id[3]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3,  _dB, _dBv4_off[3]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[4])  && WidthInRange(_inW_id[4])  && (_inN_id[4]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 4,  _dB, _dBv4_off[4]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[5])  && WidthInRange(_inW_id[5])  && (_inN_id[5]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 5,  _dB, _dBv4_off[5]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[6])  && WidthInRange(_inW_id[6])  && (_inN_id[6]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 6,  _dB, _dBv4_off[6]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[7])  && WidthInRange(_inW_id[7])  && (_inN_id[7]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 7,  _dB, _dBv4_off[7]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[8])  && WidthInRange(_inW_id[8])  && (_inN_id[8]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 8,  _dB, _dBv4_off[8]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[9])  && WidthInRange(_inW_id[9])  && (_inN_id[9]  < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 9,  _dB, _dBv4_off[9]);  \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[10]) && WidthInRange(_inW_id[10]) && (_inN_id[10] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 10, _dB, _dBv4_off[10]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[11]) && WidthInRange(_inW_id[11]) && (_inN_id[11] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 11, _dB, _dBv4_off[11]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[12]) && WidthInRange(_inW_id[12]) && (_inN_id[12] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 12, _dB, _dBv4_off[12]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[13]) && WidthInRange(_inW_id[13]) && (_inN_id[13] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 13, _dB, _dBv4_off[13]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[14]) && WidthInRange(_inW_id[14]) && (_inN_id[14] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 14, _dB, _dBv4_off[14]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[15]) && WidthInRange(_inW_id[15]) && (_inN_id[15] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 15, _dB, _dBv4_off[15]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[16]) && WidthInRange(_inW_id[16]) && (_inN_id[16] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 16, _dB, _dBv4_off[16]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[17]) && WidthInRange(_inW_id[17]) && (_inN_id[17] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 17, _dB, _dBv4_off[17]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[18]) && WidthInRange(_inW_id[18]) && (_inN_id[18] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 18, _dB, _dBv4_off[18]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[19]) && WidthInRange(_inW_id[19]) && (_inN_id[19] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 19, _dB, _dBv4_off[19]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[20]) && WidthInRange(_inW_id[20]) && (_inN_id[20] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 20, _dB, _dBv4_off[20]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[21]) && WidthInRange(_inW_id[21]) && (_inN_id[21] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 21, _dB, _dBv4_off[21]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[22]) && WidthInRange(_inW_id[22]) && (_inN_id[22] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 22, _dB, _dBv4_off[22]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[23]) && WidthInRange(_inW_id[23]) && (_inN_id[23] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 23, _dB, _dBv4_off[23]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[24]) && WidthInRange(_inW_id[24]) && (_inN_id[24] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 24, _dB, _dBv4_off[24]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[25]) && WidthInRange(_inW_id[25]) && (_inN_id[25] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 25, _dB, _dBv4_off[25]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[26]) && WidthInRange(_inW_id[26]) && (_inN_id[26] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 26, _dB, _dBv4_off[26]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[27]) && WidthInRange(_inW_id[27]) && (_inN_id[27] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 27, _dB, _dBv4_off[27]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[28]) && WidthInRange(_inW_id[28]) && (_inN_id[28] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 28, _dB, _dBv4_off[28]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[29]) && WidthInRange(_inW_id[29]) && (_inN_id[29] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 29, _dB, _dBv4_off[29]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[30]) && WidthInRange(_inW_id[30]) && (_inN_id[30] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 30, _dB, _dBv4_off[30]); \
            CP_ASYNC( ( fltCv16Valid && HeightInRange(_inH_id[31]) && WidthInRange(_inW_id[31]) && (_inN_id[31] < inNum) ), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 31, _dB, _dBv4_off[31]); \
        }
