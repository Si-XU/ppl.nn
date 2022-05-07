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
#elif defined(ENABLE_FUSE)
__global__ void __launch_bounds__(CTA_SIZE_IN_THD) KERNEL_NAME(TOTAL_KPARAM_LIST)
#endif
{
#if (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__  * 1000  + __CUDACC_VER_MINOR__ * 10  >= 10020)
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
    //  warp layout
    uint warp_idx  = (tid >>  WARP_SIZE_IN_BITS) & (CTA_SIZE_X_IN_WARP - 1);
    uint warp_idy  =  tid >> (WARP_SIZE_IN_BITS  +  CTA_SIZE_X_IN_BITS);

    uint ldg_idx   =  tid % TILE_K_V16_PER_CTA;
    uint ldg_idy   =  tid / TILE_K_V16_PER_CTA;

#if TILE_K_PER_CTA == 32
    uint sts_idx   = ((tid & 0x1) ^ ( (tid & 0xf) >> 3));
    uint sts_idy   =   tid >> 1;
#elif TILE_K_PER_CTA == 64
    uint sts_idx   = ((tid & 0x3) ^ ( (tid & 0x1f) >> 3));
    uint sts_idy   =   tid >> 2;
#elif TILE_K_PER_CTA == 128
    uint sts_idx   = ((tid & 0x7) ^ ( (tid & 0x3f) >> 3));
    uint sts_idy   =   tid >> 3;
#endif

    uint out_tid   =  warp_idy * WARP_SIZE_IN_THD + local_tid;

    uint cta_idx   = blockIdx.x;
    uint cta_idy   = blockIdx.y;

#if defined(ENABLE_SPLITK)
    uint grp_id    = blockIdx.z % numGrp;
    uint spk_id    = blockIdx.z / numGrp;

    uint numChlPerSpk = (spk_id != splitK - 1) ? numChlPerSpkHead : numChlPerSpkTail;
#elif defined(ENABLE_FUSE)
    uint grp_id    = blockIdx.z;
#endif

#if defined(ENABLE_SPLITK)
    int kloop = fltHW * CeilDiv(numChlPerSpk, TILE_K_PER_CTA);
#elif defined(ENABLE_FUSE)
    int kloop = kLoopNum;
#endif

    /////////////////////////
    // in, flt and out leading dimension

    uint numChlPerGrpPadV16 = numChlPerGrpPad >> 4;
    // uint numFltPerGrpPadV16 = numFltPerGrpPad >> 4;
    uint numFltPerGrpPadV4  = numFltPerGrpPad >> 2;

    /////////////////////////
    //  output layout

    uint   dCv4_idx[OUTPUT_BLKS_PER_STEP];
    bool dCv4XValid[OUTPUT_BLKS_PER_STEP];

    uint dCv4_idy   =  cta_idy  * TILE_M_V4_PER_CTA  +
                       out_tid  % TILE_M_V4_PER_CTA;

    dCv4_idx[0]     =  cta_idx  * TILE_N_V1_PER_CTA  +
                       warp_idx * TILE_N_V1_PER_WARP +
                       out_tid  / TILE_M_V4_PER_CTA;

    bool dCv4YValid =  (dCv4_idy < numFltPerGrpPadV4) & ((out_tid / TILE_M_V4_PER_CTA) < TILE_N_PER_MMA);

#pragma unroll
    for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++)
    {
        dCv4_idx[i]   =  dCv4_idx[0] + OUTPUT_SIZE_X_IN_THD * i;
        dCv4XValid[i] = (dCv4_idx[i] / outHW) < inNum;
    }

#if defined(ENABLE_SPLITK)
    uint dCv4_base  =  spk_id * numFltPerGrpPadV4 * numGrp * outHW * inNum +
                       grp_id * numFltPerGrpPadV4 + dCv4_idy;
#elif defined(ENABLE_FUSE)
    uint dCv4_base  =  grp_id * numFltPerGrpPadV4 + dCv4_idy;
#endif

    /////////////////////////
    //  reduce layout

    uint mma_idy    =  local_tid %  MMA_SIZE_Y_IN_THD;
    uint mma_idx    =  local_tid >> MMA_SIZE_Y_IN_BITS;

    uint sRowWt_id  =  (warp_idy * TILE_M_V8_PER_WARP) / SMEM_ROW_V8_SIZE;
    uint sRowWt_off = ((warp_idy * TILE_M_V8_PER_WARP) ^ ((mma_idx % TILE_N_PER_MMA_QTR) / M_ROWS_PER_SMEM_ROW)
                       ) % SMEM_ROW_V8_SIZE;

    uint sRv2_write =  warp_idx   * TILE_M_V2_PER_CTA    * TILE_N_V1_PER_MMA    +
                       mma_idx    * TILE_M_V2_PER_CTA    +
                       sRowWt_id  * SMEM_ROW_V2_SIZE     +
                       mma_idy;

    uint sMmaRd_idy =  out_tid    % TILE_M_V4_PER_CTA;
    uint sMmaRd_idx =  out_tid    / TILE_M_V4_PER_CTA; 

    uint sRowRd_id  =  sMmaRd_idy / SMEM_ROW_V4_SIZE;
    uint sRowRd_off =  sMmaRd_idy % SMEM_ROW_V4_SIZE;

    uint sIntra_off =  sRowRd_off % TILE_M_V4_PER_MMA;
    uint sInter_off =  sRowRd_off / TILE_M_V4_PER_MMA;

    uint sRv4_read  =  warp_idx                          * TILE_M_V4_PER_CTA    * TILE_N_PER_MMA          +
                      (sMmaRd_idx / TILE_N_PER_MMA_QTR)  * TILE_M_V4_PER_CTA    * TILE_N_PER_MMA_QTR      +
                      (sMmaRd_idx % TILE_N_PER_MMA_QTR)  * TILE_M_V4_PER_CTA    +
                       sRowRd_id  * SMEM_ROW_V4_SIZE     +
                    (((sMmaRd_idx % TILE_N_PER_MMA_QTR)  / M_ROWS_PER_SMEM_ROW) ^ sInter_off)         * TILE_M_V4_PER_MMA +
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
    int  fltCv16End   = (spk_id * numChlPerSpkHead + numChlPerSpk) >> 4;
    int  fltCv16_id   = ldg_idx + ((spk_id * numChlPerSpkHead) >> 4);
#elif defined(ENABLE_FUSE)
    int  fltCv16End   = numChlPerGrpPadV16;
    int  fltCv16_id   = ldg_idx;
#endif

    bool fltCv16Valid = fltCv16_id < fltCv16End;

    /////////////////////////
    // ldg A and B registers
    int4 Rv4[Rv4_SIZE];
#if defined(ENABLE_FUSE)
    int * R = (int *) Rv4;
    float * fR = (float *) Rv4;
#endif

#if BUF_NUM <= 2
    const int4 ZEROv4 = {0, 0, 0, 0};

    int4 * reg_dAv4 = (int4 *) Rv4;
    int4 * reg_dBv4 = (int4 *) Rv4 + REG_dAv4_SIZE;
#endif

    /////////////////////////
    // ldg A index

    int   dAv4_off[READ_dAv4_STEPS];
    bool fltNValid[READ_dAv4_STEPS];

    for(int i = 0; i < READ_dAv4_STEPS; i++)
    {
        SET_dAv4_BOUND(i, dAv4_off[i], fltNValid[i]);
    }

    /////////////////////////
    // ldg B index

#if defined(FLT_SIZE1)
    int   dBv4_off[READ_dBv4_STEPS];
    bool inHWValid[READ_dBv4_STEPS];

    for(int i = 0; i < READ_dBv4_STEPS; i++)
    {
        SET_dBv4_BOUND(i, dBv4_off[i], inHWValid[i]);
    }
#elif defined(FLT_SIZE3)
    int dBv4_off[READ_dBv4_STEPS];
    int inHWMask[READ_dBv4_STEPS];

    for(int i = 0; i < READ_dBv4_STEPS; i++)
    {
        SET_dBv4_BOUND(i, dBv4_off[i], inHWMask[i]);
    }
#elif defined(FLT_SIZEN)
    int dBv4_off[READ_dBv4_STEPS];
    int   inN_id[READ_dBv4_STEPS];
    int   inH_id[READ_dBv4_STEPS];
    int   inW_id[READ_dBv4_STEPS];

    int inH_START[READ_dBv4_STEPS];
    int inW_START[READ_dBv4_STEPS];

    for(int i = 0; i < READ_dBv4_STEPS; i++)
    {
        SET_dBv4_BOUND(i, dBv4_off[i], inN_id[i], inH_START[i], inW_START[i]);
        inH_id[i] = inH_START[i];
        inW_id[i] = inW_START[i];
    }
#endif

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
    uint sAv4_write =  sts_idy  * TILE_K_V16_PER_CTA +                 // row offset
                       sts_idx;                                        // col offset

    uint sBv4_write =  sAv4_write + SM_A_V4_1BUF * BUF_NUM;

    uint ldsa_idy = ((local_tid & 0x10) >> 1) + (local_tid & 0x7);
    uint ldsb_idy =   local_tid & 0xf;

#if TILE_K_PER_CTA == 32
    uint ldsa_idx = ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x1) ^ ((local_tid & 0x8) >> 3);
    uint ldsb_idx = ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x1) ^  (local_tid >> 4);
#elif TILE_K_PER_CTA == 64
    uint ldsa_idx = ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x3) ^ ((local_tid & 0x8) >> 3);
    uint ldsb_idx = ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x3) ^  (local_tid >> 4);
#elif TILE_K_PER_CTA == 128
    uint ldsa_idx = ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x7) ^ ((local_tid & 0x8) >> 3);
    uint ldsb_idx = ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x7) ^  (local_tid >> 4);
#endif

    // load from shared memory (lds)
    uint sAv1_read  =  warp_idy   * TILE_M_PER_WARP        * TILE_K_V4_PER_CTA +
#if TILE_M_PER_WARP == 8
                      (ldsa_idy   % WARP_SIZE_IN_THD_HALF) * TILE_K_V4_PER_CTA +
#elif TILE_M_PER_WARP == 16 || TILE_M_PER_WARP == 32 || TILE_M_PER_WARP == 64
                       ldsa_idy   * TILE_K_V4_PER_CTA      +
#endif
                       ldsa_idx   * _INT4_TO_4INT_;

    uint sBv1_read  =  warp_idx   * TILE_N_PER_WARP        * TILE_K_V4_PER_CTA +
                       ldsb_idy   * TILE_K_V4_PER_CTA      +
                       ldsb_idx   * _INT4_TO_4INT_         +
                       SM_A_V1_1BUF * BUF_NUM;

#if BUF_NUM > 2
    int sm_write_buf = 0;
    int sm_read_buf  = 0;
#endif

    // double buffer registers
    int db0_sBv1[REG_sBv1_SIZE];
#if TILE_K_PER_CTA == 64 || TILE_K_PER_CTA == 128
    int db1_sBv1[REG_sBv1_SIZE];
#endif

    int db0_sAv1[REG_sAv1_SIZE];
#if TILE_K_PER_CTA == 64 || TILE_K_PER_CTA == 128
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
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, fltCv16Valid, fltNValid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, fltCv16Valid, inHWValid);
#elif BUF_NUM > 2
        LOAD_dAv4(smp_base_v4, sAv4_write, dA, dAv4_off, fltCv16Valid, fltNValid);
        LOAD_dBv4(smp_base_v4, sBv4_write, dB, dBv4_off, fltCv16Valid, inHWValid);
#endif

        FWD_FLT(fltCv16_id, fltCv16Valid);
#elif defined(FLT_SIZE3)
#if BUF_NUM <=2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, fltCv16Valid, fltNValid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, fltCv16Valid, fltHW_bid);
#elif BUF_NUM > 2
        LOAD_dAv4(smp_base_v4, sAv4_write, dA, dAv4_off, fltCv16Valid, fltNValid);
        LOAD_dBv4(smp_base_v4, sBv4_write, dB, dBv4_off, fltCv16Valid, fltHW_bid);
#endif

        FWD_FLT(fltHW_id, fltHW_bid, fltCv16_id, fltCv16Valid);
        FWD_LUT(cLut_id);

#elif defined(FLT_SIZEN)
#if BUF_NUM <=2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, fltCv16Valid, fltNValid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, inN_id, inH_id, inW_id);
#elif BUF_NUM > 2
        LOAD_dAv4(smp_base_v4, sAv4_write, dA, dAv4_off, fltCv16Valid, fltNValid);
        LOAD_dBv4(smp_base_v4, sBv4_write, dB, dBv4_off, inN_id, inH_id, inW_id);
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

#if TILE_K_PER_CTA == 64 || TILE_K_PER_CTA == 128
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
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, fltCv16Valid, fltNValid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, fltCv16Valid, inHWValid);
#elif BUF_NUM > 2
        LOAD_dAv4(smp_base_v4, sAv4_write, dA, dAv4_off, fltCv16Valid, fltNValid);
        LOAD_dBv4(smp_base_v4, sBv4_write, dB, dBv4_off, fltCv16Valid, inHWValid);
#endif

        FWD_FLT(fltCv16_id, fltCv16Valid);
#elif defined(FLT_SIZE3)
#if BUF_NUM <= 2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, fltCv16Valid, fltNValid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, fltCv16Valid, fltHW_bid);
#elif BUF_NUM > 2
        LOAD_dAv4(smp_base_v4, sAv4_write, dA, dAv4_off, fltCv16Valid, fltNValid);
        LOAD_dBv4(smp_base_v4, sBv4_write, dB, dBv4_off, fltCv16Valid, fltHW_bid);
#endif

        FWD_FLT(fltHW_id, fltHW_bid, fltCv16_id, fltCv16Valid);
        FWD_LUT(cLut_id);
#elif defined(FLT_SIZEN)
#if BUF_NUM <= 2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, fltCv16Valid, fltNValid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, inN_id, inH_id, inW_id);
#elif BUF_NUM > 2
        LOAD_dAv4(smp_base_v4, sAv4_write, dA, dAv4_off, fltCv16Valid, fltNValid);
        LOAD_dBv4(smp_base_v4, sBv4_write, dB, dBv4_off, inN_id, inH_id, inW_id);
#endif

        FWD_FLT(fltH_id, fltW_id, fltCv16_id, fltCv16Valid);
        FWD_LUT(cLut_id);
#endif

#if BUF_NUM > 2
        FWD_BUF(sm_write_buf, sAv4_write, SM_A_V4_1BUF, sBv4_write, SM_B_V4_1BUF);
#endif

#if TILE_K_PER_CTA == 64 || TILE_K_PER_CTA == 128
        // 1st step
        READ_sAv1(db1_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db1_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP2(sAv1_read);
        FWD_KGROUP_STEP2(sBv1_read);
#endif

        MMA_INSTS(C, db0_sBv1, db0_sAv1);

#if TILE_K_PER_CTA == 128
        // 2nd step
        READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP3(sAv1_read);
        FWD_KGROUP_STEP3(sBv1_read);

        MMA_INSTS(C, db1_sBv1, db1_sAv1);

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

#if TILE_K_PER_CTA == 64
        MMA_INSTS(C, db1_sBv1, db1_sAv1);
#elif TILE_K_PER_CTA == 128
        MMA_INSTS(C, db0_sBv1, db0_sAv1);
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

        // 1th step
        READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);

#if TILE_K_PER_CTA == 64 || TILE_K_PER_CTA == 128
        FWD_KGROUP_STEP1(sAv1_read);
        FWD_KGROUP_STEP1(sBv1_read);
#endif

#if TILE_K_PER_CTA == 128
        MMA_INSTS(C, db1_sBv1, db1_sAv1);
#endif
    }

#if BUF_NUM > 2
    CP_ASYNC_FENSE();
    CP_ASYNC_WAIT_ALL();
#endif
    __syncthreads();

    ///////////////////////////////////////////////////
    // output section
    ///////////////////////////////////////////////////

#if defined(ENABLE_FUSE)
    uint concatV4_off[OUTPUT_BLKS_PER_STEP];

#pragma unroll
    for (int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) { concatV4_off[i] = 0; }

    float4 deScaleV4;
    float* deScale = (float *) &deScaleV4;
    GET_DEQUANTSCALE(deScaleV4, deScale, dFltScale, inScale);

#endif

#pragma unroll
    for(int s = 0; s < OUTPUT_STEPS; s++)
    {
        WRITE_sRv2(sm_base_v2, sRv2_write, Cv2, s * BLK_N_PER_MMA * NUM_M_STEPS);

        __syncthreads();

        READ_sRv4(Rv4, sm_base_v4, sRv4_read);

        __syncthreads();

#if defined(ENABLE_FUSE)
        DEQUANT_V4(fR, R, deScale);
#endif

#if defined(ENABLE_FUSE)
        ADD_BIAS_V4(hasBias, bias);
#endif

#if defined(ENABLE_FUSE)

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
        OUTPUT_BY_INT8_V4(R);
#elif defined(ENABLE_SPLITK)
        OUTPUT_BY_INT4_V1(Rv4);
#endif
    }

#endif // __CUDA_ARCH__
}
