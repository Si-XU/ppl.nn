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

#define LOAD_dBv4_SIZE_16TH(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE_8TH(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE_QTR(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE_HALF(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE1(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            \
            CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE2(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            _dBv4_off[1] += cFltLut.idx[cLut_id]; \
            \
            CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]); \
            CP_ASYNC( (_fltNValid[1] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]); \
        }

#define LOAD_dBv4_SIZE4(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            _dBv4_off[0] += cFltLut.idx[cLut_id]; \
            _dBv4_off[1] += cFltLut.idx[cLut_id]; \
            _dBv4_off[2] += cFltLut.idx[cLut_id]; \
            _dBv4_off[3] += cFltLut.idx[cLut_id]; \
            \
            CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]); \
            CP_ASYNC( (_fltNValid[1] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]); \
            CP_ASYNC( (_fltNValid[2] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2, _dB, _dBv4_off[2]); \
            CP_ASYNC( (_fltNValid[3] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3, _dB, _dBv4_off[3]); \
        }

#define LOAD_dBv4_SIZE8(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
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
            CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]); \
            CP_ASYNC( (_fltNValid[1] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]); \
            CP_ASYNC( (_fltNValid[2] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2, _dB, _dBv4_off[2]); \
            CP_ASYNC( (_fltNValid[3] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3, _dB, _dBv4_off[3]); \
            CP_ASYNC( (_fltNValid[4] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 4, _dB, _dBv4_off[4]); \
            CP_ASYNC( (_fltNValid[5] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 5, _dB, _dBv4_off[5]); \
            CP_ASYNC( (_fltNValid[6] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 6, _dB, _dBv4_off[6]); \
            CP_ASYNC( (_fltNValid[7] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 7, _dB, _dBv4_off[7]); \
        }

#define LOAD_dBv4_SIZE16(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
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
            CP_ASYNC( (_fltNValid[0]  && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]);  \
            CP_ASYNC( (_fltNValid[1]  && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]);  \
            CP_ASYNC( (_fltNValid[2]  && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2, _dB, _dBv4_off[2]);  \
            CP_ASYNC( (_fltNValid[3]  && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3, _dB, _dBv4_off[3]);  \
            CP_ASYNC( (_fltNValid[4]  && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 4, _dB, _dBv4_off[4]);  \
            CP_ASYNC( (_fltNValid[5]  && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 5, _dB, _dBv4_off[5]);  \
            CP_ASYNC( (_fltNValid[6]  && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 6, _dB, _dBv4_off[6]);  \
            CP_ASYNC( (_fltNValid[7]  && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 7, _dB, _dBv4_off[7]);  \
            CP_ASYNC( (_fltNValid[8]  && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 8, _dB, _dBv4_off[8]);  \
            CP_ASYNC( (_fltNValid[9]  && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 9, _dB, _dBv4_off[9]);  \
            CP_ASYNC( (_fltNValid[10] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 10, _dB, _dBv4_off[10]); \
            CP_ASYNC( (_fltNValid[11] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 11, _dB, _dBv4_off[11]); \
            CP_ASYNC( (_fltNValid[12] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 12, _dB, _dBv4_off[12]); \
            CP_ASYNC( (_fltNValid[13] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 13, _dB, _dBv4_off[13]); \
            CP_ASYNC( (_fltNValid[14] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 14, _dB, _dBv4_off[14]); \
            CP_ASYNC( (_fltNValid[15] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 15, _dB, _dBv4_off[15]); \
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

#define SET_dAv4_BOUND(_step_id, _dAv4_off, _inHWMask) \
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
            _dAv4_off  =  (_inN_id  * inHW + _inH_id  * inWidth + _inW_id) * numChlPerGrpPadV16 * numGrp + \
                           grp_id   * numChlPerGrpPadV16 + \
                           fltCv16_id; \
            \
            _inH_id =  _inH_id - padHeight; \
            _inW_id =  _inW_id - padWidth;  \
            \
            SET_BOUND_FLT3(_inHWMask, _inN_id, _inH_id, _inW_id); \
        }

#define LOAD_dAv4_SIZE_16TH(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                CP_ASYNC( ((_fltHW_bid & inHWMask[0]) && _inCv16Valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE_8TH(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                CP_ASYNC( ((_fltHW_bid & inHWMask[0]) && _inCv16Valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE_QTR(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                CP_ASYNC( ((_fltHW_bid & inHWMask[0]) && _inCv16Valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE_HALF(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                CP_ASYNC( ((_fltHW_bid & inHWMask[0]) && _inCv16Valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE1(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            \
            CP_ASYNC( ((_fltHW_bid & inHWMask[0]) && _inCv16Valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE2(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            _dAv4_off[1] += cInLut.idx[cLut_id]; \
            \
            CP_ASYNC( ((_fltHW_bid & inHWMask[0]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[1]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1, _dA, _dAv4_off[1]); \
        }

#define LOAD_dAv4_SIZE4(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _fltHW_bid) \
        { \
            _dAv4_off[0] += cInLut.idx[cLut_id]; \
            _dAv4_off[1] += cInLut.idx[cLut_id]; \
            _dAv4_off[2] += cInLut.idx[cLut_id]; \
            _dAv4_off[3] += cInLut.idx[cLut_id]; \
            \
            CP_ASYNC( ((_fltHW_bid & inHWMask[0]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[1]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1, _dA, _dAv4_off[1]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[2]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2, _dA, _dAv4_off[2]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[3]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3, _dA, _dAv4_off[3]); \
        }

#define LOAD_dAv4_SIZE8(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _fltHW_bid) \
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
            CP_ASYNC( ((_fltHW_bid & inHWMask[0]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[1]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1, _dA, _dAv4_off[1]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[2]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2, _dA, _dAv4_off[2]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[3]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3, _dA, _dAv4_off[3]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[4]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 4, _dA, _dAv4_off[4]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[5]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 5, _dA, _dAv4_off[5]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[6]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 6, _dA, _dAv4_off[6]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[7]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 7, _dA, _dAv4_off[7]); \
        }

#define LOAD_dAv4_SIZE16(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _fltHW_bid) \
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
            CP_ASYNC( ((_fltHW_bid & inHWMask[0])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0,  _dA, _dAv4_off[0]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[1])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1,  _dA, _dAv4_off[1]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[2])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2,  _dA, _dAv4_off[2]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[3])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3,  _dA, _dAv4_off[3]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[4])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 4,  _dA, _dAv4_off[4]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[5])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 5,  _dA, _dAv4_off[5]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[6])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 6,  _dA, _dAv4_off[6]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[7])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 7,  _dA, _dAv4_off[7]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[8])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 8,  _dA, _dAv4_off[8]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[9])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 9,  _dA, _dAv4_off[9]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[10]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 10, _dA, _dAv4_off[10]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[11]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 11, _dA, _dAv4_off[11]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[12]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 12, _dA, _dAv4_off[12]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[13]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 13, _dA, _dAv4_off[13]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[14]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 14, _dA, _dAv4_off[14]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[15]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 15, _dA, _dAv4_off[15]); \
        }

#define LOAD_dAv4_SIZE32(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _fltHW_bid) \
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
            CP_ASYNC( ((_fltHW_bid & inHWMask[0])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0,  _dA, _dAv4_off[0]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[1])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1,  _dA, _dAv4_off[1]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[2])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2,  _dA, _dAv4_off[2]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[3])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3,  _dA, _dAv4_off[3]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[4])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 4,  _dA, _dAv4_off[4]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[5])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 5,  _dA, _dAv4_off[5]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[6])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 6,  _dA, _dAv4_off[6]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[7])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 7,  _dA, _dAv4_off[7]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[8])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 8,  _dA, _dAv4_off[8]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[9])  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 9,  _dA, _dAv4_off[9]);  \
            CP_ASYNC( ((_fltHW_bid & inHWMask[10]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 10, _dA, _dAv4_off[10]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[11]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 11, _dA, _dAv4_off[11]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[12]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 12, _dA, _dAv4_off[12]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[13]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 13, _dA, _dAv4_off[13]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[14]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 14, _dA, _dAv4_off[14]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[15]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 15, _dA, _dAv4_off[15]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[16]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 16, _dA, _dAv4_off[16]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[17]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 17, _dA, _dAv4_off[17]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[18]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 18, _dA, _dAv4_off[18]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[19]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 19, _dA, _dAv4_off[19]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[20]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 20, _dA, _dAv4_off[20]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[21]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 21, _dA, _dAv4_off[21]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[22]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 22, _dA, _dAv4_off[22]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[23]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 23, _dA, _dAv4_off[23]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[24]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 24, _dA, _dAv4_off[24]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[25]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 25, _dA, _dAv4_off[25]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[26]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 26, _dA, _dAv4_off[26]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[27]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 27, _dA, _dAv4_off[27]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[28]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 28, _dA, _dAv4_off[28]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[29]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 29, _dA, _dAv4_off[29]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[30]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 30, _dA, _dAv4_off[30]); \
            CP_ASYNC( ((_fltHW_bid & inHWMask[31]) && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 31, _dA, _dAv4_off[31]); \
        }
