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
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE_8TH(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE_QTR(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE_HALF(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE1(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE2(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]); \
            CP_ASYNC( (_fltNValid[1] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]); \
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
            _dBv4_off[1] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE4(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]); \
            CP_ASYNC( (_fltNValid[1] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]); \
            CP_ASYNC( (_fltNValid[2] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2, _dB, _dBv4_off[2]); \
            CP_ASYNC( (_fltNValid[3] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3, _dB, _dBv4_off[3]); \
            \
            _dBv4_off[0] += TILE_K_V16_PER_CTA; \
            _dBv4_off[1] += TILE_K_V16_PER_CTA; \
            _dBv4_off[2] += TILE_K_V16_PER_CTA; \
            _dBv4_off[3] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dBv4_SIZE8(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
            CP_ASYNC( (_fltNValid[0] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]); \
            CP_ASYNC( (_fltNValid[1] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]); \
            CP_ASYNC( (_fltNValid[2] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2, _dB, _dBv4_off[2]); \
            CP_ASYNC( (_fltNValid[3] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3, _dB, _dBv4_off[3]); \
            CP_ASYNC( (_fltNValid[4] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 4, _dB, _dBv4_off[4]); \
            CP_ASYNC( (_fltNValid[5] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 5, _dB, _dBv4_off[5]); \
            CP_ASYNC( (_fltNValid[6] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 6, _dB, _dBv4_off[6]); \
            CP_ASYNC( (_fltNValid[7] && _fltCv16Valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 7, _dB, _dBv4_off[7]); \
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

#define LOAD_dBv4_SIZE16(_sBv4, _sBv4_off, _dB, _dBv4_off, _fltCv16Valid, _fltNValid) \
        { \
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
            _fltNValid  =  _fltN_id < numFltPerGrpPad; \
            \
            _dBv4_off  =   grp_id   * numChlPerGrpPadV16 * fltHW  * numFltPerGrpPad + \
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

#define LOAD_dAv4_SIZE_16TH(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                CP_ASYNC( (_inHWValid[0] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE_8TH(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                CP_ASYNC( (_inHWValid[0] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE_QTR(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                CP_ASYNC( (_inHWValid[0] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE_HALF(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                CP_ASYNC( (_inHWValid[0] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE1(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            CP_ASYNC( (_inHWValid[0] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE2(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            CP_ASYNC( (_inHWValid[0] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            CP_ASYNC( (_inHWValid[1] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1, _dA, _dAv4_off[1]); \
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
            _dAv4_off[1] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE4(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            CP_ASYNC( (_inHWValid[0] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            CP_ASYNC( (_inHWValid[1] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1, _dA, _dAv4_off[1]); \
            CP_ASYNC( (_inHWValid[2] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2, _dA, _dAv4_off[2]); \
            CP_ASYNC( (_inHWValid[3] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3, _dA, _dAv4_off[3]); \
            \
            _dAv4_off[0] += TILE_K_V16_PER_CTA; \
            _dAv4_off[1] += TILE_K_V16_PER_CTA; \
            _dAv4_off[2] += TILE_K_V16_PER_CTA; \
            _dAv4_off[3] += TILE_K_V16_PER_CTA; \
        }

#define LOAD_dAv4_SIZE8(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            CP_ASYNC( (_inHWValid[0] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            CP_ASYNC( (_inHWValid[1] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1, _dA, _dAv4_off[1]); \
            CP_ASYNC( (_inHWValid[2] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2, _dA, _dAv4_off[2]); \
            CP_ASYNC( (_inHWValid[3] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3, _dA, _dAv4_off[3]); \
            CP_ASYNC( (_inHWValid[4] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 4, _dA, _dAv4_off[4]); \
            CP_ASYNC( (_inHWValid[5] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 5, _dA, _dAv4_off[5]); \
            CP_ASYNC( (_inHWValid[6] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 6, _dA, _dAv4_off[6]); \
            CP_ASYNC( (_inHWValid[7] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 7, _dA, _dAv4_off[7]); \
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

#define LOAD_dAv4_SIZE16(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            CP_ASYNC( (_inHWValid[0]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0,  _dA, _dAv4_off[0]);  \
            CP_ASYNC( (_inHWValid[1]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1,  _dA, _dAv4_off[1]);  \
            CP_ASYNC( (_inHWValid[2]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2,  _dA, _dAv4_off[2]);  \
            CP_ASYNC( (_inHWValid[3]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3,  _dA, _dAv4_off[3]);  \
            CP_ASYNC( (_inHWValid[4]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 4,  _dA, _dAv4_off[4]);  \
            CP_ASYNC( (_inHWValid[5]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 5,  _dA, _dAv4_off[5]);  \
            CP_ASYNC( (_inHWValid[6]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 6,  _dA, _dAv4_off[6]);  \
            CP_ASYNC( (_inHWValid[7]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 7,  _dA, _dAv4_off[7]);  \
            CP_ASYNC( (_inHWValid[8]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 8,  _dA, _dAv4_off[8]);  \
            CP_ASYNC( (_inHWValid[9]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 9,  _dA, _dAv4_off[9]);  \
            CP_ASYNC( (_inHWValid[10] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 10, _dA, _dAv4_off[10]); \
            CP_ASYNC( (_inHWValid[11] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 11, _dA, _dAv4_off[11]); \
            CP_ASYNC( (_inHWValid[12] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 12, _dA, _dAv4_off[12]); \
            CP_ASYNC( (_inHWValid[13] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 13, _dA, _dAv4_off[13]); \
            CP_ASYNC( (_inHWValid[14] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 14, _dA, _dAv4_off[14]); \
            CP_ASYNC( (_inHWValid[15] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 15, _dA, _dAv4_off[15]); \
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

#define LOAD_dAv4_SIZE32(_sAv4, _sAv4_off, _dA, _dAv4_off, _inCv16Valid, _inHWValid) \
        { \
            CP_ASYNC( (_inHWValid[0]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0,  _dA, _dAv4_off[0]);  \
            CP_ASYNC( (_inHWValid[1]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1,  _dA, _dAv4_off[1]);  \
            CP_ASYNC( (_inHWValid[2]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2,  _dA, _dAv4_off[2]);  \
            CP_ASYNC( (_inHWValid[3]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3,  _dA, _dAv4_off[3]);  \
            CP_ASYNC( (_inHWValid[4]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 4,  _dA, _dAv4_off[4]);  \
            CP_ASYNC( (_inHWValid[5]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 5,  _dA, _dAv4_off[5]);  \
            CP_ASYNC( (_inHWValid[6]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 6,  _dA, _dAv4_off[6]);  \
            CP_ASYNC( (_inHWValid[7]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 7,  _dA, _dAv4_off[7]);  \
            CP_ASYNC( (_inHWValid[8]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 8,  _dA, _dAv4_off[8]);  \
            CP_ASYNC( (_inHWValid[9]  && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 9,  _dA, _dAv4_off[9]);  \
            CP_ASYNC( (_inHWValid[10] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 10, _dA, _dAv4_off[10]); \
            CP_ASYNC( (_inHWValid[11] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 11, _dA, _dAv4_off[11]); \
            CP_ASYNC( (_inHWValid[12] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 12, _dA, _dAv4_off[12]); \
            CP_ASYNC( (_inHWValid[13] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 13, _dA, _dAv4_off[13]); \
            CP_ASYNC( (_inHWValid[14] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 14, _dA, _dAv4_off[14]); \
            CP_ASYNC( (_inHWValid[15] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 15, _dA, _dAv4_off[15]); \
            CP_ASYNC( (_inHWValid[16] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 16, _dA, _dAv4_off[16]); \
            CP_ASYNC( (_inHWValid[17] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 17, _dA, _dAv4_off[17]); \
            CP_ASYNC( (_inHWValid[18] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 18, _dA, _dAv4_off[18]); \
            CP_ASYNC( (_inHWValid[19] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 19, _dA, _dAv4_off[19]); \
            CP_ASYNC( (_inHWValid[20] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 20, _dA, _dAv4_off[20]); \
            CP_ASYNC( (_inHWValid[21] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 21, _dA, _dAv4_off[21]); \
            CP_ASYNC( (_inHWValid[22] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 22, _dA, _dAv4_off[22]); \
            CP_ASYNC( (_inHWValid[23] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 23, _dA, _dAv4_off[23]); \
            CP_ASYNC( (_inHWValid[24] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 24, _dA, _dAv4_off[24]); \
            CP_ASYNC( (_inHWValid[25] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 25, _dA, _dAv4_off[25]); \
            CP_ASYNC( (_inHWValid[26] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 26, _dA, _dAv4_off[26]); \
            CP_ASYNC( (_inHWValid[27] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 27, _dA, _dAv4_off[27]); \
            CP_ASYNC( (_inHWValid[28] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 28, _dA, _dAv4_off[28]); \
            CP_ASYNC( (_inHWValid[29] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 29, _dA, _dAv4_off[29]); \
            CP_ASYNC( (_inHWValid[30] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 30, _dA, _dAv4_off[30]); \
            CP_ASYNC( (_inHWValid[31] && _inCv16Valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 31, _dA, _dAv4_off[31]); \
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
