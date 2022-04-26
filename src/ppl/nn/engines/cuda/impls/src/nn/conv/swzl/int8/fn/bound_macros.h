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

#define FWD_FLT_SIZE1(_fltH_id, _fltW_id, _fltCv16_id, _fltCv16Valid) \
        { \
            _fltW_id++; \
            inW_id[0] += holeWidth; \
            \
            if(_fltW_id == fltWidth) \
            {\
                _fltW_id = 0; \
                inW_id[0] = inW_START[0]; \
                _fltH_id++; \
                inH_id[0] += holeHeight; \
            } \
            \
            if(_fltH_id == fltHeight) \
            { \
                _fltH_id = 0;   \
                inH_id[0] = inH_START[0]; \
                \
                _fltCv16_id += TILE_K_V16_PER_CTA; \
                \
                _fltCv16Valid = _fltCv16_id < fltCv16End; \
            } \
        }

#define FWD_FLT_SIZE2(_fltH_id, _fltW_id, _fltCv16_id, _fltCv16Valid) \
        { \
            _fltW_id++; \
            inW_id[0] += holeWidth;        inW_id[1] += holeWidth; \
            \
            if(_fltW_id == fltWidth) \
            {\
                _fltW_id = 0; \
                inW_id[0] = inW_START[0];  inW_id[1] = inW_START[1]; \
                _fltH_id++; \
                inH_id[0] += holeHeight;   inH_id[1] += holeHeight; \
            } \
            \
            if(_fltH_id == fltHeight) \
            { \
                _fltH_id = 0;   \
                inH_id[0] = inH_START[0];  inH_id[1] = inH_START[1]; \
                \
                _fltCv16_id += TILE_K_V16_PER_CTA; \
                \
                _fltCv16Valid = _fltCv16_id < fltCv16End; \
            } \
        }

#define FWD_FLT_SIZE4(_fltH_id, _fltW_id, _fltCv16_id, _fltCv16Valid) \
        { \
            _fltW_id++; \
            inW_id[0] += holeWidth;        inW_id[1] += holeWidth;   inW_id[2] += holeWidth;  inW_id[3] += holeWidth; \
            \
            if(_fltW_id == fltWidth) \
            { \
                _fltW_id = 0; \
                inW_id[0] = inW_START[0];  inW_id[1] = inW_START[1]; inW_id[2] = inW_START[2];  inW_id[3] = inW_START[3]; \
                _fltH_id++; \
                inH_id[0] += holeHeight;   inH_id[1] += holeHeight;  inH_id[2] += holeHeight;   inH_id[3] += holeHeight; \
            } \
            \
            if(_fltH_id == fltHeight) \
            { \
                _fltH_id = 0;   \
                inH_id[0] = inH_START[0];  inH_id[1] = inH_START[1]; inH_id[2] = inH_START[2];  inH_id[3] = inH_START[3]; \
                \
                _fltCv16_id += TILE_K_V16_PER_CTA; \
                \
                _fltCv16Valid = _fltCv16_id < fltCv16End; \
            } \
        }

#define FWD_FLT_SIZE8(_fltH_id, _fltW_id, _fltCv16_id, _fltCv16Valid) \
        { \
            _fltW_id++; \
            inW_id[0] += holeWidth;        inW_id[1] += holeWidth;   inW_id[2] += holeWidth;  inW_id[3] += holeWidth; \
            inW_id[4] += holeWidth;        inW_id[5] += holeWidth;   inW_id[6] += holeWidth;  inW_id[7] += holeWidth; \
            \
            if(_fltW_id == fltWidth) \
            { \
                _fltW_id = 0; \
                inW_id[0] = inW_START[0];  inW_id[1] = inW_START[1]; inW_id[2] = inW_START[2];  inW_id[3] = inW_START[3]; \
                inW_id[4] = inW_START[4];  inW_id[5] = inW_START[5]; inW_id[6] = inW_START[6];  inW_id[7] = inW_START[7]; \
                _fltH_id++; \
                inH_id[0] += holeHeight;   inH_id[1] += holeHeight;  inH_id[2] += holeHeight;   inH_id[3] += holeHeight; \
                inH_id[4] += holeHeight;   inH_id[5] += holeHeight;  inH_id[6] += holeHeight;   inH_id[7] += holeHeight; \
            } \
            \
            if(_fltH_id == fltHeight) \
            { \
                _fltH_id = 0;   \
                inH_id[0] = inH_START[0];  inH_id[1] = inH_START[1]; inH_id[2] = inH_START[2];  inH_id[3] = inH_START[3]; \
                inH_id[4] = inH_START[4];  inH_id[5] = inH_START[5]; inH_id[6] = inH_START[6];  inH_id[7] = inH_START[7]; \
                \
                _fltCv16_id += TILE_K_V16_PER_CTA; \
                \
                _fltCv16Valid = _fltCv16_id < fltCv16End; \
            } \
        }

#define FWD_FLT_SIZE16(_fltH_id, _fltW_id, _fltCv16_id, _fltCv16Valid) \
        { \
            _fltW_id++; \
            inW_id[0]  += holeWidth;        inW_id[1]  += holeWidth;   inW_id[2]  += holeWidth;  inW_id[3]  += holeWidth; \
            inW_id[4]  += holeWidth;        inW_id[5]  += holeWidth;   inW_id[6]  += holeWidth;  inW_id[7]  += holeWidth; \
            inW_id[8]  += holeWidth;        inW_id[9]  += holeWidth;   inW_id[10] += holeWidth;  inW_id[11] += holeWidth; \
            inW_id[12] += holeWidth;        inW_id[13] += holeWidth;   inW_id[14] += holeWidth;  inW_id[15] += holeWidth; \
            \
            if(_fltW_id == fltWidth) \
            { \
                _fltW_id = 0; \
                inW_id[0]  = inW_START[0];   inW_id[1]  = inW_START[1];  inW_id[2]  = inW_START[2];   inW_id[3]  = inW_START[3]; \
                inW_id[4]  = inW_START[4];   inW_id[5]  = inW_START[5];  inW_id[6]  = inW_START[6];   inW_id[7]  = inW_START[7]; \
                inW_id[8]  = inW_START[8];   inW_id[9]  = inW_START[9];  inW_id[10] = inW_START[10];  inW_id[11] = inW_START[11]; \
                inW_id[12] = inW_START[12];  inW_id[13] = inW_START[13]; inW_id[14] = inW_START[14];  inW_id[15] = inW_START[15]; \
                _fltH_id++; \
                inH_id[0]  += holeHeight;        inH_id[1]  += holeHeight;   inH_id[2]  += holeHeight;  inH_id[3]  += holeHeight; \
                inH_id[4]  += holeHeight;        inH_id[5]  += holeHeight;   inH_id[6]  += holeHeight;  inH_id[7]  += holeHeight; \
                inH_id[8]  += holeHeight;        inH_id[9]  += holeHeight;   inH_id[10] += holeHeight;  inH_id[11] += holeHeight; \
                inH_id[12] += holeHeight;        inH_id[13] += holeHeight;   inH_id[14] += holeHeight;  inH_id[15] += holeHeight; \
            } \
            \
            if(_fltH_id == fltHeight) \
            { \
                _fltH_id = 0;   \
                inH_id[0]  = inH_START[0];   inH_id[1]  = inH_START[1];  inH_id[2]  = inH_START[2];   inH_id[3]  = inH_START[3]; \
                inH_id[4]  = inH_START[4];   inH_id[5]  = inH_START[5];  inH_id[6]  = inH_START[6];   inH_id[7]  = inH_START[7]; \
                inH_id[8]  = inH_START[8];   inH_id[9]  = inH_START[9];  inH_id[10] = inH_START[10];  inH_id[11] = inH_START[11]; \
                inH_id[12] = inH_START[12];  inH_id[13] = inH_START[13]; inH_id[14] = inH_START[14];  inH_id[15] = inH_START[15]; \
                \
                _fltCv16_id += TILE_K_V16_PER_CTA; \
                \
                _fltCv16Valid = _fltCv16_id < fltCv16End; \
            } \
        }
