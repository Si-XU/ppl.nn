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

#define SET_BOUND_FLT3(_inHWMask, _inN_id, _inH_id, _inW_id) \
        { \
            if(_inN_id < inNum) \
            { \
                _inHWMask = 0xffffffff; \
                if(_inH_id < 0 || _inH_id >= inHeight) _inHWMask = _inHWMask & 0xfffffff8; \
                if(_inW_id < 0 || _inW_id >= inWidth)  _inHWMask = _inHWMask & 0xffffffb6; \
                \
                _inH_id += holeHeight; \
                _inW_id += holeWidth; \
                \
                if(_inH_id < 0 || _inH_id >= inHeight) _inHWMask = _inHWMask & 0xffffffc7; \
                if(_inW_id < 0 || _inW_id >= inWidth)  _inHWMask = _inHWMask & 0xffffff6d; \
                \
                _inH_id += holeHeight; \
                _inW_id += holeWidth; \
                \
                if(_inH_id < 0 || _inH_id >= inHeight)  _inHWMask = _inHWMask & 0xfffffe3f; \
                if(_inW_id < 0 || _inW_id >= inWidth)   _inHWMask = _inHWMask & 0xfffffedb; \
            } else { \
                _inHWMask = 0x0; \
            } \
        }

#define FWD_FLT3(_fltHW_id, _fltHW_bid, _fltCv16_id, _fltCv16Valid) \
        { \
            if(_fltHW_id == 8) \
            { \
                _fltHW_id = 0; \
                _fltCv16_id += TILE_K_V16_PER_CTA; \
                \
                _fltCv16Valid = _fltCv16_id < fltCv16End; \
            } else { \
                _fltHW_id = _fltHW_id + 1; \
            } \
            \
            _fltHW_bid = (0x1 << _fltHW_id); \
        }

#define FWD_FLT(_flt_hw_id, _flt_hw_bid, _flt_c_v16_id, _flt_c_v16_valid) FWD_FLT3(_flt_hw_id, _flt_hw_bid, _flt_c_v16_id, _flt_c_v16_valid)
