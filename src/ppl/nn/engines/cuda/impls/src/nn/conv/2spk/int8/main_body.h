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

#if defined(ENABLE_SPLITK)
__global__ void __launch_bounds__(CTA_SIZE_IN_THD) KERNEL_NAME(SPK_KPARAM_LIST)
#elif defined(ENABLE_FUSE) || defined(ENABLE_SPLITF)
__global__ void __launch_bounds__(CTA_SIZE_IN_THD) KERNEL_NAME(TOTAL_KPARAM_LIST)
#endif
{
#if (__CUDA_ARCH__ >= 750) && (__CUDACC_VER_MAJOR__  * 1000  + __CUDACC_VER_MINOR__ * 10  >= 10020)
    ///////////////////////////////////////////////////
    // definition section
    ///////////////////////////////////////////////////

    /////////////////////////
    // results
    int4 Cv4[Cv4_ITEMS_PER_THD];

    int  * C   = (int  *) Cv4;
    int2 * Cv2 = (int2 *) Cv4;

#pragma unroll
    for (int i = 0; i < C_ITEMS_PER_THD; i++) { C[i] = _ZERO_; }

    /////////////////////////
    // thread layout
    uint tid       =  threadIdx.x;

    uint local_tid =  tid & 0x1f; // [0, 31], tid within the warp

    /////////////////////////
    //  set layout

    uint set_tid   =  tid & (SET_SIZE_IN_THD - 1);

    uint set_id    = (tid >> SET_SIZE_IN_BITS) & 0x7; // at most 8 set is feasible

    uint set_widx  = (set_tid >>  WARP_SIZE_IN_BITS) & (SET_SIZE_X_IN_WARP - 1);
    uint set_widy  =  set_tid >> (WARP_SIZE_IN_BITS  +  SET_SIZE_X_IN_BITS);
 
    uint ldg_idx   =  tid % TILE_K_V16_PER_CTA;
    uint ldg_idy   =  tid / TILE_K_V16_PER_CTA;

#if TILE_K_PER_CTA == 16
    uint sts_idx   =   0;
    uint sts_idy   =   tid;
#elif TILE_K_PER_CTA == 32
    uint sts_idx   = ((tid & 0x1) ^ ( (tid & 0xf) >> 3));
    uint sts_idy   =   tid >> 1;
#elif TILE_K_PER_CTA == 64
    uint sts_idx   = ((tid & 0x3) ^ ( (tid & 0x1f) >> 3));
    uint sts_idy   =   tid >> 2;
#elif TILE_K_PER_CTA == 128
    uint sts_idx   = ((tid & 0x7) ^ ( (tid & 0x3f) >> 3));
    uint sts_idy   =   tid >> 3;
#elif TILE_K_PER_CTA == 256
    uint sts_idx   = ((tid & 0xf) ^ ( (tid & 0x7f) >> 4));
    uint sts_idy   =   tid >> 4;
#endif

    /////////////////////////
    //  cta layout
    uint cta_idx   = blockIdx.y;
    uint cta_idy   = blockIdx.x;

#if defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)
    uint spk_id    =  blockIdx.z %  splitK;
    uint spf_id    = (blockIdx.z % (splitK * fltHW)) / splitK;
    uint grp_id    =  blockIdx.z / (splitK * fltHW);

    uint numChlPerSpk = (spk_id != splitK - 1) ? numChlPerSpkHead : numChlPerSpkTail;
#elif defined(ENABLE_SPLITK) && !defined(ENABLE_SPLITF)
    uint spk_id    =  blockIdx.z %  splitK;
    uint grp_id    =  blockIdx.z /  splitK;

    uint numChlPerSpk = (spk_id != splitK - 1) ? numChlPerSpkHead : numChlPerSpkTail;
#elif !defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)
    uint spf_id    =  blockIdx.z %  fltHW;
    uint grp_id    =  blockIdx.z /  fltHW;
#elif defined(ENABLE_FUSE)
    uint grp_id    = blockIdx.z;
#endif

#if defined(ENABLE_SPLITK)
    int kloop = fltHW * CeilDiv(numChlPerSpk, TILE_K_PER_SET);
#elif defined(ENABLE_SPLITF) || defined(ENABLE_FUSE)
    int kloop = kLoopNum;
#endif

    /////////////////////////
    // in, flt and out leading dimension

    uint numChlPerGrpPadV16 = numChlPerGrpPad >> 4;
    // uint numFltPerGrpPadV16 = numFltPerGrpPad >> 4;
    uint numFltPerGrpPadV4  = numFltPerGrpPad >> 2;

    /////////////////////////
    //  output layout

    uint dCv4_idy   =  cta_idy  * TILE_M_V1_PER_CTA  +
                       tid      / TILE_N_V4_PER_CTA;

    uint dCv4_idx   =  cta_idx  * TILE_N_V4_PER_CTA  +
                       tid      % TILE_N_V4_PER_CTA;

    bool dCv4XValid =  (dCv4_idx < numFltPerGrpPadV4) & ((tid / TILE_N_V4_PER_CTA) < TILE_M_PER_CTA);

#if defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)
    uint dCv4_base  = (spf_id   * splitK + spk_id)  * numFltPerGrpPadV4 * numGrp * outHW * inNum +
                       grp_id   * numFltPerGrpPadV4;
#elif defined(ENABLE_SPLITK) && !defined(ENABLE_SPLITF)
    uint dCv4_base  =  spk_id   * numFltPerGrpPadV4 * numGrp * outHW * inNum +
                       grp_id   * numFltPerGrpPadV4;
#elif !defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)
    uint dCv4_base  =  spf_id   * numFltPerGrpPadV4 * numGrp * outHW * inNum +
                       grp_id   * numFltPerGrpPadV4;
#elif defined(ENABLE_FUSE)
    uint dCv4_base  =  grp_id   * numFltPerGrpPadV4;
#endif

    /////////////////////////
    //  reduce layout

    uint mma_idx    =  local_tid %  MMA_SIZE_X_IN_THD;
    uint mma_idy    =  local_tid >> MMA_SIZE_X_IN_BITS;

    uint sRowWt_id  =  (set_widx * TILE_N_V8_PER_WARP) / SMEM_ROW_V8_SIZE;
#if (SET_SIZE_Y_IN_WARP * INTER_SET_REDUCE_RATIO * WARP_SIZE_IN_THD / TILE_N_V4_PER_WARP) == 2
#if defined(USE_IMMA16816)
    uint sRowWt_off = ((set_widx * TILE_N_V8_PER_WARP) ^ ((mma_idy % TILE_M_PER_MMA_8TH) / N_ROWS_PER_SMEM_ROW)
#elif defined(USE_IMMA8816)
    uint sRowWt_off = ((set_widx * TILE_N_V8_PER_WARP) ^ ((mma_idy % TILE_M_PER_MMA_QTR) / N_ROWS_PER_SMEM_ROW)
#endif
#else
#if defined(USE_IMMA16816)
    uint sRowWt_off = ((set_widx * TILE_N_V8_PER_WARP) ^ ((mma_idy % TILE_M_PER_MMA_QTR) / N_ROWS_PER_SMEM_ROW)
#elif defined(USE_IMMA8816)
    uint sRowWt_off = ((set_widx * TILE_N_V8_PER_WARP) ^ ((mma_idy % TILE_M_PER_MMA_HALF) / N_ROWS_PER_SMEM_ROW)
#endif
#endif
                       ) % SMEM_ROW_V8_SIZE;

    uint sRv2_write =  set_id     * TILE_N_V2_PER_CTA    * TILE_M_V1_PER_CTA  +
                       set_widy   * TILE_N_V2_PER_CTA    * TILE_M_V1_PER_WARP +
                       mma_idy    * TILE_N_V2_PER_CTA    +
                       sRowWt_id  * SMEM_ROW_V2_SIZE     +
                       mma_idx;

    uint sMmaRd_idx =  tid        % TILE_N_V4_PER_CTA;
    uint sMmaRd_idy =  tid        / TILE_N_V4_PER_CTA; 

    uint sRowRd_id  =  sMmaRd_idx / SMEM_ROW_V4_SIZE;
    uint sRowRd_off =  sMmaRd_idx % SMEM_ROW_V4_SIZE;

    uint sIntra_off =  sRowRd_off % TILE_N_V4_PER_MMA;
    uint sInter_off =  sRowRd_off / TILE_N_V4_PER_MMA;

#if defined(USE_IMMA16816)
    uint sRv4_read  = (sMmaRd_idy / TILE_M_PER_MMA_QTR)  * TILE_N_V4_PER_CTA    * TILE_M_PER_MMA_QTR  +
                      (sMmaRd_idy % TILE_M_PER_MMA_QTR)  * TILE_N_V4_PER_CTA    +
                       sRowRd_id  * SMEM_ROW_V4_SIZE     +
                    (((sMmaRd_idy % TILE_M_PER_MMA_QTR)  / N_ROWS_PER_SMEM_ROW) ^ sInter_off)         * TILE_N_V4_PER_MMA +
#elif defined(USE_IMMA8816)
    uint sRv4_read  = (sMmaRd_idy / TILE_M_PER_MMA_HALF) * TILE_N_V4_PER_CTA    * TILE_M_PER_MMA_HALF +
                      (sMmaRd_idy % TILE_M_PER_MMA_HALF) * TILE_N_V4_PER_CTA    +
                       sRowRd_id  * SMEM_ROW_V4_SIZE     +
                    (((sMmaRd_idy % TILE_M_PER_MMA_HALF) / N_ROWS_PER_SMEM_ROW) ^ sInter_off)         * TILE_N_V4_PER_MMA +
#endif
                       sIntra_off;

    ///////////////////////////////////////////////////
    // device memory A B and C index
    ///////////////////////////////////////////////////

#if defined(FLT_SIZE3)
    int fltHW_id      = 0;
    int fltHW_bid     = 0x1; // bid: binary id for bit operation

    int cLut_id       = 0;
#elif defined(FLT_SIZEN)
    int  fltH_id      = 0;
    int  fltW_id      = 0;

    int cLut_id       = 0;
#endif

#if defined(ENABLE_SPLITK)
    int  fltCv16End    = (spk_id * numChlPerSpkHead + numChlPerSpk) >> 4;
    int  fltCv16_id    = ldg_idx + ((spk_id * numChlPerSpkHead) >> 4);
#elif defined(ENABLE_SPLITF) || defined(ENABLE_FUSE)
    int  fltCv16End    = numChlPerGrpPadV16;
    int  fltCv16_id    = ldg_idx;
#endif

    bool fltCv16Valid  = fltCv16_id < fltCv16End;

    /////////////////////////
    // ldg A and B registers
    int4 Rv4[Rv4_SIZE];
#if defined(ENABLE_FUSE) || ((defined(ENABLE_SPLITF) || defined(ENABLE_SPLITK)) && (TILE_K_PER_CTA > TILE_K_PER_SET))
    int * R = (int *) Rv4;
#endif
#if defined(ENABLE_FUSE)
    float * fR = (float *) Rv4;
#endif

#if BUF_NUM <= 2
    const int4 ZEROv4 = {0, 0, 0, 0};

    int4 * reg_dAv4 = (int4 *) Rv4;
    int4 * reg_dBv4 = (int4 *) Rv4 + REG_dAv4_SIZE;
#endif

    /////////////////////////
    // ldg A index

#if defined(FLT_SIZE1)
    int   dAv4_off[READ_dAv4_STEPS];
    bool inHWValid[READ_dAv4_STEPS];

#pragma unroll
    for(int i = 0; i < READ_dAv4_STEPS; i++)
    {
        SET_dAv4_BOUND(i, dAv4_off[i], inHWValid[i]);
    }
#elif defined(FLT_SIZE3)
    int dAv4_off[READ_dAv4_STEPS];
    int inHWMask[READ_dAv4_STEPS];

#pragma unroll
    for(int i = 0; i < READ_dAv4_STEPS; i++)
    {
        SET_dAv4_BOUND(i, dAv4_off[i], inHWMask[i]);
    }
#elif defined(FLT_SIZEN)
    int dAv4_off[READ_dAv4_STEPS];
    int   inN_id[READ_dAv4_STEPS];
    int   inH_id[READ_dAv4_STEPS];
    int   inW_id[READ_dAv4_STEPS];

    int inH_START[READ_dAv4_STEPS];
    int inW_START[READ_dAv4_STEPS];

#pragma unroll
    for(int i = 0; i < READ_dAv4_STEPS; i++)
    {
        SET_dAv4_BOUND(i, dAv4_off[i], inN_id[i], inH_START[i], inW_START[i]);
        inH_id[i] = inH_START[i];
        inW_id[i] = inW_START[i];
    }
#endif

    /////////////////////////
    // ldg B index

    int   dBv4_off[READ_dBv4_STEPS];
    bool fltNValid[READ_dBv4_STEPS];

#pragma unroll
    for(int i = 0; i < READ_dBv4_STEPS; i++)
    {
        SET_dBv4_BOUND(i, dBv4_off[i], fltNValid[i]);
    }

    ///////////////////////////////////////////////////
    // shared memory index
    ///////////////////////////////////////////////////

    /////////////////////////
    //  smem index
    extern __shared__ char sm_base[];

    int  * sm_base_v1 = (int  *) sm_base;
    int2 * sm_base_v2 = (int2 *) sm_base;
    int4 * sm_base_v4 = (int4 *) sm_base;

    uint32_t smp_base_v1;
    CVT_SM_PTR(smp_base_v1, sm_base_v1);

#if BUF_NUM > 2
    uint32_t smp_base_v4;
    CVT_SM_PTR(smp_base_v4, sm_base_v4);
#endif

    /////////////////////////
    //  A, B smem index

    // store to shared memory (sts)
    uint sAv4_write =  sts_idy  * TILE_K_V16_PER_CTA +                  // row offset
                       sts_idx;                                         // col offset

    uint sBv4_write =  sAv4_write + SM_A_V4_1BUF * BUF_NUM;

    uint lds_idy =  local_tid;
#if TILE_K_PER_CTA == 16
    uint lds_idx =  0;
#elif TILE_K_PER_CTA == 32
    uint lds_idx = ((set_id * TILE_K_V16_PER_SET) & 0x1) ^ ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x1);
#elif TILE_K_PER_CTA == 64
    uint lds_idx = ((set_id * TILE_K_V16_PER_SET) & 0x3) ^ ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x3);
#elif TILE_K_PER_CTA == 128
    uint lds_idx = ((set_id * TILE_K_V16_PER_SET) & 0x7) ^ ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x7);
#elif TILE_K_PER_CTA == 256
    uint lds_idx = ((set_id * TILE_K_V16_PER_SET) & 0xf) ^ (local_tid & 0x7);
#endif

    // load from shared memory (lds)
    uint sAv1_read  =  set_widy   * TILE_M_PER_WARP        * TILE_K_V4_PER_CTA +
#if TILE_M_PER_WARP == 8
                      (lds_idy    % WARP_SIZE_IN_THD_QTR)  * TILE_K_V4_PER_CTA +
#elif TILE_M_PER_WARP == 16
                      (lds_idy    % WARP_SIZE_IN_THD_HALF) * TILE_K_V4_PER_CTA +
#elif TILE_M_PER_WARP == 32
                       lds_idy    * TILE_K_V4_PER_CTA      +
#elif TILE_M_PER_WARP == 64 || TILE_M_PER_WARP == 128
                       lds_idy    * TILE_K_V4_PER_CTA      +
#endif
                       lds_idx    * _INT4_TO_4INT_;

    uint sBv1_read  =  set_widx   * TILE_N_PER_WARP        * TILE_K_V4_PER_CTA +
#if TILE_N_PER_WARP == 8
                      (lds_idy    % WARP_SIZE_IN_THD_QTR)  * TILE_K_V4_PER_CTA +
#elif TILE_N_PER_WARP == 16
                      (lds_idy    % WARP_SIZE_IN_THD_HALF) * TILE_K_V4_PER_CTA +
#elif TILE_N_PER_WARP == 32 || TILE_N_PER_WARP == 64
                       lds_idy    * TILE_K_V4_PER_CTA      +
#endif
                       lds_idx    * _INT4_TO_4INT_         +
                       SM_A_V1_1BUF * BUF_NUM;

#if BUF_NUM > 2
    int sm_write_buf = 0;
    int sm_read_buf  = 0;
#endif

    // double buffer registers
    int db0_sBv1[REG_sBv1_SIZE];
#if TILE_K_PER_SET == 32 || TILE_K_PER_SET == 64
    int db1_sBv1[REG_sBv1_SIZE];
#endif

    int db0_sAv1[REG_sAv1_SIZE];
#if TILE_K_PER_SET == 32 || TILE_K_PER_SET == 64
    int db1_sAv1[REG_sAv1_SIZE];
#endif

    ///////////////////////////////////////////////////
    // main loop
    ///////////////////////////////////////////////////

    // prefetch
#if BUF_NUM > 2
#pragma unroll
    for(int buf = 0; buf < BUF_NUM - 1; buf++, --kloop)
    {
#endif
#if defined(FLT_SIZE1)
#if BUF_NUM <=2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, fltCv16Valid, inHWValid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, fltCv16Valid, fltNValid);
#elif BUF_NUM > 2
        LOAD_dAv4(smp_base_v4, sAv4_write, dA, dAv4_off, fltCv16Valid, inHWValid);
        LOAD_dBv4(smp_base_v4, sBv4_write, dB, dBv4_off, fltCv16Valid, fltNValid);
#endif

        FWD_FLT(fltCv16_id, fltCv16Valid);
#elif defined(FLT_SIZE3)
#if BUF_NUM <=2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, fltCv16Valid, fltHW_bid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, fltCv16Valid, fltNValid);
#elif BUF_NUM > 2
        LOAD_dAv4(smp_base_v4, sAv4_write, dA, dAv4_off, fltCv16Valid, fltHW_bid);
        LOAD_dBv4(smp_base_v4, sBv4_write, dB, dBv4_off, fltCv16Valid, fltNValid);
#endif

        FWD_FLT(fltHW_id, fltHW_bid, fltCv16_id, fltCv16Valid);
        FWD_LUT(cLut_id);

#elif defined(FLT_SIZEN)
#if BUF_NUM <=2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, inN_id, inH_id, inW_id);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, fltCv16Valid, fltNValid);
#elif BUF_NUM > 2
        LOAD_dAv4(smp_base_v4, sAv4_write, dA, dAv4_off, inN_id, inH_id, inW_id);
        LOAD_dBv4(smp_base_v4, sBv4_write, dB, dBv4_off, fltCv16Valid, fltNValid);
#endif

        FWD_FLT(fltH_id, fltW_id, fltCv16_id, fltCv16Valid);
        FWD_LUT(cLut_id);
#endif

#if BUF_NUM > 2
        FWD_BUF(sm_write_buf, sAv4_write, SM_A_V4_1BUF, sBv4_write, SM_B_V4_1BUF);

        CP_ASYNC_FENSE();
    }
#endif

#if BUF_NUM <= 2
    WRITE_sAv4(sm_base_v4, sAv4_write, reg_dAv4);
    WRITE_sBv4(sm_base_v4, sBv4_write, reg_dBv4);
#elif BUF_NUM > 2
    CP_ASYNC_WAIT_ALL_BUT(BUF_NUM - INFLIGHT_BUF_NUM);
#endif
    __syncthreads();

#if BUF_NUM == 2
    FWD_BUF(sAv4_write, SM_A_V4_1BUF, 0, sBv4_write, SM_B_V4_1BUF, SM_A_V4_2BUF);
#endif

    READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);
    READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);

#if TILE_K_PER_SET == 32 || TILE_K_PER_SET == 64
    FWD_KGROUP_STEP1(sAv1_read);
    FWD_KGROUP_STEP1(sBv1_read);
#endif

#if BUF_NUM <= 2
    for (; kloop > 0; --kloop)
#elif BUF_NUM > 2
    for (; kloop > (1 - BUF_NUM); --kloop)
#endif
    {
#if defined(FLT_SIZE1)
#if BUF_NUM <= 2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, fltCv16Valid, inHWValid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, fltCv16Valid, fltNValid);
#elif BUF_NUM > 2
        LOAD_dAv4(smp_base_v4, sAv4_write, dA, dAv4_off, fltCv16Valid, inHWValid);
        LOAD_dBv4(smp_base_v4, sBv4_write, dB, dBv4_off, fltCv16Valid, fltNValid);
#endif

        FWD_FLT(fltCv16_id, fltCv16Valid);
#elif defined(FLT_SIZE3)
#if BUF_NUM <= 2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, fltCv16Valid, fltHW_bid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, fltCv16Valid, fltNValid);
#elif BUF_NUM > 2
        LOAD_dAv4(smp_base_v4, sAv4_write, dA, dAv4_off, fltCv16Valid, fltHW_bid);
        LOAD_dBv4(smp_base_v4, sBv4_write, dB, dBv4_off, fltCv16Valid, fltNValid);
#endif

        FWD_FLT(fltHW_id, fltHW_bid, fltCv16_id, fltCv16Valid);
        FWD_LUT(cLut_id);
#elif defined(FLT_SIZEN)
#if BUF_NUM <= 2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, inN_id, inH_id, inW_id);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, fltCv16Valid, fltNValid);
#elif BUF_NUM > 2
        LOAD_dAv4(smp_base_v4, sAv4_write, dA, dAv4_off, inN_id, inH_id, inW_id);
        LOAD_dBv4(smp_base_v4, sBv4_write, dB, dBv4_off, fltCv16Valid, fltNValid);
#endif

        FWD_FLT(fltH_id, fltW_id, fltCv16_id, fltCv16Valid);
        FWD_LUT(cLut_id);
#endif

#if BUF_NUM > 2
        FWD_BUF(sm_write_buf, sAv4_write, SM_A_V4_1BUF, sBv4_write, SM_B_V4_1BUF);
#endif

#if TILE_K_PER_SET == 32 || TILE_K_PER_SET == 64
        // 1st step
        READ_sAv1(db1_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db1_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP2(sAv1_read);
        FWD_KGROUP_STEP2(sBv1_read);
#endif

        MMA_INSTS(C, db0_sAv1, db0_sBv1);

#if TILE_K_PER_SET == 64
        // 2nd step
        READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP3(sAv1_read);
        FWD_KGROUP_STEP3(sBv1_read);

        MMA_INSTS(C, db1_sAv1, db1_sBv1);

        // 3rd step
        READ_sAv1(db1_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db1_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP4(sAv1_read);
        FWD_KGROUP_STEP4(sBv1_read);
#endif

#if BUF_NUM == 1
        __syncthreads();
#endif

#if BUF_NUM <= 2
        WRITE_sAv4(sm_base_v4, sAv4_write, reg_dAv4);
        WRITE_sBv4(sm_base_v4, sBv4_write, reg_dBv4);
#endif

#if TILE_K_PER_SET == 32
        MMA_INSTS(C, db1_sAv1, db1_sBv1);
#elif TILE_K_PER_SET == 64
        MMA_INSTS(C, db0_sAv1, db0_sBv1);
#endif

#if BUF_NUM == 2
        FWD_BUF(sAv4_write, SM_A_V4_1BUF, 0, sBv4_write, SM_B_V4_1BUF, SM_A_V4_2BUF);

        FWD_BUF(sAv1_read,  SM_A_V1_1BUF, 0, sBv1_read,  SM_B_V1_1BUF, SM_A_V1_2BUF);
#elif BUF_NUM > 2
        CP_ASYNC_FENSE();
        CP_ASYNC_WAIT_ALL_BUT(BUF_NUM - INFLIGHT_BUF_NUM);
        FWD_BUF(sm_read_buf, sAv1_read, SM_A_V1_1BUF, sBv1_read, SM_B_V1_1BUF);
#endif

        __syncthreads();

        // 1st step
        READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);

#if TILE_K_PER_SET == 32 || TILE_K_PER_SET == 64
        FWD_KGROUP_STEP1(sAv1_read);
        FWD_KGROUP_STEP1(sBv1_read);
#endif

#if TILE_K_PER_SET == 64
        MMA_INSTS(C, db1_sAv1, db1_sBv1);
#endif
    }

#if BUF_NUM > 2
    CP_ASYNC_FENSE();
    CP_ASYNC_WAIT_ALL();
#endif
    __syncthreads();

    ///////////////////////////////////////////////////
    // reduce section
    ///////////////////////////////////////////////////

    WRITE_sRv2(sm_base_v2, sRv2_write, Cv2);

    __syncthreads();

    ///////////////////////////////////////////////////
    // output section
    ///////////////////////////////////////////////////

#pragma unroll
    for(int s = 0; s < OUTPUT_STEPS; s++)
    {
        READ_sRv4(Rv4, sm_base_v4, sRv4_read);

#if TILE_K_PER_CTA > TILE_K_PER_SET
        REDUCE(R);
#endif
        bool dCv4YValid = (dCv4_idy / outHW) < inNum;
        uint dCv4_off = dCv4_base +
                        dCv4_idy  * numFltPerGrpPadV4 * numGrp +
                        dCv4_idx;

#if defined(ENABLE_FUSE)
        float4 deScaleV4;
        float* deScale = (float *) &deScaleV4;
        GET_DEQUANTSCALE(deScaleV4, deScale, dFltScale, inScale);
        DEQUANT_V4(fR, R, deScale);
#endif

#if defined(ENABLE_FUSE)
        ADD_BIAS_V4(hasBias, bias);
#endif

#if defined(ENABLE_FUSE)
        uint concatV4_off = 0;

        FUSE_RELU_V4(hasRelu);
        FUSE_CLIP_V4(hasClip, clipMax, clipMin);
        // FUSE_PRELU_V4(hasPrelu, prelu, leaky);

        FUSE_ELT_V4(hasElt, preData);
        FUSE_RELU_V4(hasEltRelu);
        FUSE_CLIP_V4(hasEltClip, eltClipMax, eltClipMin);
        // FUSE_PRELU_V4(hasEltPrelu, eltPrelu, eltLeaky);

        SET_CONCAT_OFF_V4(hasConcat, concatV4_off);

        QUANT_V4(R, fR, outScale);
#endif

#if defined(ENABLE_FUSE)
        PACK_V4(R, 0);
        OUTPUT_BY_INT8_V4(R);
#elif defined(ENABLE_SPLITK) || defined(ENABLE_SPLITF)
        OUTPUT_BY_INT4_V1(Rv4);
#endif

        dCv4_idy += OUTPUT_SIZE_Y_IN_THD;
    }

#endif // __CUDA_ARCH__
}
