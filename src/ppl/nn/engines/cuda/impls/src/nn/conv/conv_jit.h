#ifndef __PPLCUDA_CONV_JIT_H__
#define __PPLCUDA_CONV_JIT_H__

#define CPI_SM75_HMMA1688       8
#define CPI_SM75_IMMA8816       4
#define CPI_SM75_LDG32_L1D      4 
#define CPI_SM75_LDG64_L1D      4 
#define CPI_SM75_LDG128_L1D     8
#define CPI_SM75_LDG32_L2       4
#define CPI_SM75_LDG64_L2       8
#define CPI_SM75_LDG128_L2      16
#define CPI_SM75_LDS128         8
#define CPI_SM75_STS128         10

#define CPI_SM80_HMMA1688       4
#define CPI_SM80_HMMA16816      8
#define CPI_SM80_IMMA8816       4
#define CPI_SM80_IMMA16816      4
#define CPI_SM80_IMMA16832      8
#define CPI_SM80_LDG32_L1D      4 
#define CPI_SM80_LDG64_L1D      4 
#define CPI_SM80_LDG128_L1D     8
#define CPI_SM80_LDG32_L2       4
#define CPI_SM80_LDG64_L2       8
#define CPI_SM80_LDG128_L2      16
#define CPI_SM80_LDS128         8
#define CPI_SM80_STS128         10

#define LATENCY_SM75_HMMA1688   14
#define LATENCY_SM75_IMMA8816   10
#define LATENCY_SM75_SMEM       19
#define LATENCY_SM75_L1D_CACHE  32
#define LATENCY_SM75_L2_CACHE   188
#define LATENCY_SM75_DRAM       296

#define LATENCY_SM80_HMMA1688   14
#define LATENCY_SM80_IMMA8816   14
#define LATENCY_SM80_HMMA16816  32
#define LATENCY_SM80_IMMA16816  14
#define LATENCY_SM80_IMMA16832  32
#define LATENCY_SM80_SMEM       22
#define LATENCY_SM80_L1D_CACHE  34
#define LATENCY_SM80_L2_CACHE   200 // near:200, far:350
#define LATENCY_SM80_DRAM       360

#define PB_NUM_PER_SM           4

#include "ppl/common/types.h"
#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/common/common.h"

// sort by ascending order
bool SortByAscendScore(const std::pair<algo_param_t, float> &a, const std::pair<algo_param_t, float> &b)
{
    return (a.second < b.second);
}

// sort by descending order
bool SortByDescendScore(const std::pair<algo_param_t, float> &a, const std::pair<algo_param_t, float> &b)
{
    return (a.second > b.second);
}

__inline__ void GetHardwareInfo(
        int device_arch,
        ppl::common::datatype_t type,
        int &cpi_mma,
        int &latency_mma,
        int &cpi_ldg32_l1d,
        int &cpi_ldg64_l1d,
        int &cpi_ldg128_l1d,
        int &cpi_ldg32_l2,
        int &cpi_ldg64_l2,
        int &cpi_ldg128_l2,
        int &latency_l2_cache,
        int &latency_dram)
{
    if (device_arch == 75) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            cpi_mma = CPI_SM75_HMMA1688;
            latency_mma = LATENCY_SM75_HMMA1688;

        } else if( type == ppl::common::DATATYPE_INT8 ) {
            cpi_mma = CPI_SM75_IMMA8816;
            latency_mma = LATENCY_SM75_IMMA8816;
        }

        cpi_ldg32_l1d  = CPI_SM75_LDG32_L1D;
        cpi_ldg64_l1d  = CPI_SM75_LDG64_L1D;
        cpi_ldg128_l1d = CPI_SM75_LDG128_L1D;

        cpi_ldg32_l2  = CPI_SM75_LDG32_L2;
        cpi_ldg64_l2  = CPI_SM75_LDG64_L2;
        cpi_ldg128_l2 = CPI_SM75_LDG128_L2;

        latency_l2_cache = LATENCY_SM75_L2_CACHE;
        latency_dram = LATENCY_SM75_DRAM;

    } else if (device_arch >= 80) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            cpi_mma = CPI_SM80_HMMA16816;
            latency_mma = LATENCY_SM80_HMMA16816;

        } else if( type == ppl::common::DATATYPE_INT8 ) {
            cpi_mma = CPI_SM80_IMMA16832;
            latency_mma = LATENCY_SM80_IMMA16832;
        }
        cpi_ldg32_l1d  = CPI_SM80_LDG32_L1D;
        cpi_ldg64_l1d  = CPI_SM80_LDG64_L1D;
        cpi_ldg128_l1d = CPI_SM80_LDG128_L1D;

        cpi_ldg32_l2  = CPI_SM80_LDG32_L2;
        cpi_ldg64_l2  = CPI_SM80_LDG64_L2;
        cpi_ldg128_l2 = CPI_SM80_LDG128_L2;

        latency_l2_cache = LATENCY_SM80_L2_CACHE;
        latency_dram = LATENCY_SM80_DRAM;
    }
}

__inline__ void GetIdxnMmaInfo(
        int device_arch,
        ppl::common::datatype_t type,
        std::string &mma_shape,
        int &m_mma,
        int &n_mma,
        int &k_mma,
        int &m_mma_ceil,
        int &n_mma_ceil,
        int &k_mma_ceil)
{
    if (device_arch == 75) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            mma_shape = "hmma1688";
            m_mma = 16;
            n_mma = 8;
            k_mma = 8;
            k_mma_ceil = k_mma * 4;
        } else if( type == ppl::common::DATATYPE_INT8 ) {
            mma_shape = "imma8816";
            m_mma = 8;
            n_mma = 8;
            k_mma = 16;
            k_mma_ceil = k_mma * 4;
        }
    } else if (device_arch >= 80) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            mma_shape = "hmma16816";
            m_mma = 16;
            n_mma = 8;
            k_mma = 16;
            k_mma_ceil = k_mma * 2;

        } else if( type == ppl::common::DATATYPE_INT8 ) {
            mma_shape = "imma16832";
            m_mma = 16;
            n_mma = 8;
            k_mma = 32;
            k_mma_ceil = k_mma * 2;
        }
    }

    m_mma_ceil = m_mma * 8;
    n_mma_ceil = n_mma * 8;
}

__inline__ void Get2spkMmaInfo(
        int device_arch,
        ppl::common::datatype_t type,
        std::string &mma_shape,
        int &m_mma,
        int &n_mma,
        int &k_mma,
        int &m_mma_ceil,
        int &n_mma_ceil,
        int &k_mma_ceil,
        int &buf_num_ceil)
{
    if (device_arch == 75) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            mma_shape = "hmma1688";
            m_mma = 16;
            n_mma = 8;
            k_mma = 8;
            buf_num_ceil = 2;
        } else if( type == ppl::common::DATATYPE_INT8 ) {
            mma_shape = "imma8816";
            m_mma = 8;
            n_mma = 8;
            k_mma = 16;
            buf_num_ceil = 2;
        }
    } else if (device_arch >= 80) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            mma_shape = "hmma16816";
            m_mma = 16;
            n_mma = 8;
            k_mma = 16;
            buf_num_ceil = 6;
        } else if( type == ppl::common::DATATYPE_INT8 ) {
            mma_shape = "imma16832";
            m_mma = 16;
            n_mma = 8;
            k_mma = 32;
            buf_num_ceil = 6;
        }
    }

    m_mma_ceil = m_mma * 4;
    n_mma_ceil = n_mma * 4;
    k_mma_ceil = k_mma * 4;
}

__inline__ void GetSwzlMmaInfo(
        int device_arch,
        ppl::common::datatype_t type,
        std::string &mma_shape,
        int &m_mma,
        int &n_mma,
        int &k_mma,
        int &m_mma_ceil,
        int &n_mma_ceil,
        int &k_mma_ceil,
        int &buf_num_ceil)
{
    if (device_arch == 75) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            mma_shape = "hmma1688";
            m_mma = 8;
            n_mma = 16;
            k_mma = 8;
            buf_num_ceil = 2;
            k_mma_ceil = k_mma * 8;
        } else if( type == ppl::common::DATATYPE_INT8 ) {
            mma_shape = "imma8816";
            m_mma = 8;
            n_mma = 8;
            k_mma = 16;
            buf_num_ceil = 2;
            k_mma_ceil = k_mma * 4;
        }
    } else if (device_arch >= 80) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            mma_shape = "hmma16816";
            m_mma = 8;
            n_mma = 16;
            k_mma = 16;
            buf_num_ceil = 6;
            k_mma_ceil = k_mma * 4;
        } else if( type == ppl::common::DATATYPE_INT8 ) {
            mma_shape = "imma16832";
            m_mma = 8;
            n_mma = 16;
            k_mma = 32;
            buf_num_ceil = 6;
            k_mma_ceil = k_mma * 4;
        }
    }

    m_mma_ceil = m_mma * 8;
    n_mma_ceil = n_mma * 8;
}

__inline__ float GetEfficiencyScore(
        int m_cta,
        int n_cta,
        int k_cta,
        int m_conv,
        int n_conv,
        int k_conv)
{
    int workload_conv   = m_conv * n_conv * k_conv;
    int workload_kernel = m_cta * DivUp(m_conv, m_cta) * \
                          n_cta * DivUp(n_conv, n_cta) * \
                          k_cta * DivUp(k_conv, k_cta);

    float eff_score = 1.0 * workload_conv / workload_kernel;

    return eff_score;
}

__inline__ int CheckIdxnSmemFeasible(
        int m_cta,
        int cta_size_in_thd)
{
    int smem_per_cta = (m_cta + cta_size_in_thd) * _INT_TO_4BYTE_; // in byte
    
    return smem_per_cta;
}

int GetIdxnRegsPerThread(
        ppl::common::datatype_t type,
        int m_cta,
        int n_cta,
        int m_warp,
        int n_warp,
        int k_per_step,
        int m_mma,
        int n_mma,
        int k_mma,
        int cta_size_in_thd)
{
    int m_blk_num = m_warp / 8;
    int n_blk_num = n_warp / 8;

    int regs_a_v1 = m_blk_num * (k_per_step / k_mma);
    int regs_b_v1 = n_blk_num * (k_per_step / k_mma);

    int regs_c_v1 = DivUp(m_cta * n_cta, cta_size_in_thd * GetPadSize(type) / 4);

    int regs_a_idx = (m_blk_num + 1) * 4;
    int regs_b_idx =  n_blk_num * 2;
    int regs_c_idx =  n_blk_num * 2  + 4;

    int regs_idx = Max(regs_a_idx + regs_b_idx, regs_c_idx);

    int regs_common = 20;

    int regs_per_thd = regs_a_v1 + regs_b_v1 + regs_c_v1 + regs_idx + regs_common;

    return regs_per_thd;
}

__inline__ float GetWarpOccupySMScore( int warp_num_per_sm)
{
    if(warp_num_per_sm >= 0 && warp_num_per_sm < 4)
        return 0.5;
    else if(warp_num_per_sm >= 4 && warp_num_per_sm < 8)
        return 0.8;
    else if(warp_num_per_sm >= 8 && warp_num_per_sm < 12)
        return 1;
    else if(warp_num_per_sm >= 12 && warp_num_per_sm < 16)
        return 0.8;
    else // if(warp_num_per_sm >= 16)
        return 0.6;

    // if(warp_num_per_sm >= 10)
    //     return (1 - (1.f * (warp_num_per_sm - 10) / 100));
    // else
    //     return (1 - (1.f * (10 - warp_num_per_sm) / 100));
}

__inline__ float GetOccupancyScore(
        int cta_size_in_thd,
        int cta_size_in_warp,
        int sm_num,
        int cta_num,
        int regs_per_cta,
        int smem_per_cta,
        int max_ctas_per_sm,
        int max_thds_per_sm,
        int max_regs_per_sm,
        int max_smem_per_sm,
        float& cta_launch_times)
{
    int cta_num_limit_by_thds = max_thds_per_sm / cta_size_in_thd;
    int cta_num_limit_by_ctas = max_regs_per_sm / regs_per_cta;
    int cta_num_limit_by_smem = max_smem_per_sm / smem_per_cta; 

    int cta_num_per_sm      = Min(max_ctas_per_sm, Min(cta_num_limit_by_thds, Min(cta_num_limit_by_ctas, cta_num_limit_by_smem)));
    int cta_num_per_launch  = cta_num_per_sm * sm_num;

    int warp_num_per_sm     = cta_num_per_sm * cta_size_in_warp;
    // int warp_num_per_launch = warp_num_per_sm * sm_num;

    int max_cta_num_on_sm = 0;
    int min_cta_num_on_sm = 0;
    
    int max_warp_num_on_sm = 0;
    int min_warp_num_on_sm = 0;

    int sm_num_of_max_occupy = 0;
    int sm_num_of_min_occupy = 0;

    // float cta_launch_times = 1.f * cta_num / cta_num_per_launch;
    cta_launch_times = 1.f * cta_num / cta_num_per_launch;

    if (cta_launch_times >= 1) { // multiple launches
        max_cta_num_on_sm  = cta_num_per_sm;
        min_cta_num_on_sm  = cta_num_per_sm;

        max_warp_num_on_sm = warp_num_per_sm;
        min_warp_num_on_sm = warp_num_per_sm;
    } else { // less than 1 launch
        max_cta_num_on_sm  = DivUp(cta_num, sm_num);
        min_cta_num_on_sm  = cta_num / sm_num;

        max_warp_num_on_sm = max_cta_num_on_sm * cta_size_in_warp;
        min_warp_num_on_sm = min_cta_num_on_sm * cta_size_in_warp;
    }

    if(cta_launch_times >= 1 || max_cta_num_on_sm == min_cta_num_on_sm) {
        sm_num_of_max_occupy = sm_num;
        sm_num_of_min_occupy = 0;
    } else {
        // max_cta_num_on_sm * sm_num_of_max_occupy + min_cta_num_on_sm * (sm_num - sm_num_of_max_occupy) = cta_num
        sm_num_of_max_occupy = (cta_num - min_cta_num_on_sm * sm_num) / (max_cta_num_on_sm - min_cta_num_on_sm);
        sm_num_of_min_occupy = sm_num - sm_num_of_max_occupy;
    }

    float sm_num_of_max_occupy_pct = 1.f * sm_num_of_max_occupy / sm_num;
    float sm_num_of_min_occupy_pct = 1.f * sm_num_of_min_occupy / sm_num;

    float score_sm_occupy = sm_num_of_max_occupy_pct * GetWarpOccupySMScore(max_warp_num_on_sm) + sm_num_of_min_occupy_pct * GetWarpOccupySMScore(min_warp_num_on_sm);

    // float score_sm_tail = 1.f * (cta_num % sm_num) / sm_num;

    // float factor_launch = Min(2.f / cta_launch_times, 0.9);

    // float score_occ = (1 - factor_launch) * score_sm_tail + factor_launch * score_sm_occupy;
    float score_occ = score_sm_occupy;

    return score_occ;
}

float GetIdxnPipelineScore(
        int type_size,
        float cta_launch_times,
        int out_w,
        int cta_size_in_thd,
        int cta_size_in_warp,
        int m_cta,
        int n_cta,
        int k_cta,
        int m_warp,
        int n_warp,
        int k_per_step,
        int m_mma,
        int n_mma,
        int k_mma,
        int cpi_mma,
        int cpi_ldg32_l1d,
        int cpi_ldg64_l1d,
        int cpi_ldg128_l1d,
        int cpi_ldg32_l2,
        int cpi_ldg64_l2,
        int cpi_ldg128_l2,
        int latency_mma,
        int latency_l2_cache,
        int latency_dram
        )
{
    int warp_num_per_pb = DivUp(cta_size_in_warp, PB_NUM_PER_SM);

    int cycles_mma = cpi_mma * (m_warp / m_mma) * (n_warp / n_mma) * (k_per_step/ k_mma) * warp_num_per_pb + latency_mma;

    int cycles_mem = 0;

    int mr_flt_total = 0;
    int mr_flt_l2  = 0;
    int mr_flt_l1d = 0;

    int mr_input_total = 0;
    int mr_input_l2 = 0;
    int mr_input_l1d = 0;
    
    if(k_per_step == 8) {
        mr_flt_total = DivUp(n_cta * k_per_step * type_size, _INT_TO_4BYTE_ * WARP_SIZE);
        mr_flt_l2  = mr_flt_total;
        mr_flt_l1d = 0;

        mr_input_total = DivUp(m_cta * k_per_step * type_size, _INT_TO_4BYTE_  * WARP_SIZE);
        mr_input_l2 = DivUp(DivUp(m_cta, out_w) * Min(out_w, m_cta) * (k_per_step >> 2) * type_size, _INT_TO_4BYTE_  * WARP_SIZE);
        mr_input_l1d = mr_input_total - mr_input_l2;

        cycles_mem = cpi_ldg32_l1d * (mr_flt_l1d + mr_input_l1d) + cpi_ldg32_l2 * (mr_flt_l2 + mr_input_l2) + latency_l2_cache;
    }
    else if(k_per_step == 16) {
        mr_flt_total = DivUp(n_cta * k_per_step * type_size, _INT2_TO_8BYTE_ * WARP_SIZE);
        mr_flt_l2  = mr_flt_total;
        mr_flt_l1d = 0;

        mr_input_total = DivUp(m_cta * k_per_step * type_size, _INT2_TO_8BYTE_  * WARP_SIZE);
        mr_input_l2 = DivUp(DivUp(m_cta, out_w) * Min(out_w, m_cta) * (k_per_step >> 2) * type_size, _INT2_TO_8BYTE_  * WARP_SIZE);
        mr_input_l1d = mr_input_total - mr_input_l2;

        cycles_mem = cpi_ldg64_l1d * (mr_flt_l1d + mr_input_l1d) + cpi_ldg64_l2 * (mr_flt_l2 + mr_input_l2) + latency_l2_cache;
    }
    else if(k_per_step == 32) {
        mr_flt_total = DivUp(n_cta * k_per_step * type_size, _INT4_TO_16BYTE_ * WARP_SIZE);
        mr_flt_l2  = mr_flt_total;
        mr_flt_l1d = 0;

        mr_input_total = DivUp(m_cta * k_per_step * type_size, _INT4_TO_16BYTE_  * WARP_SIZE);
        mr_input_l2 = DivUp(DivUp(m_cta, out_w) * Min(out_w, m_cta) * (k_per_step >> 2) * type_size, _INT4_TO_16BYTE_  * WARP_SIZE);
        mr_input_l1d = mr_input_total - mr_input_l2;

        cycles_mem = cpi_ldg128_l1d * (mr_flt_l1d + mr_input_l1d) + cpi_ldg128_l2 * (mr_flt_l2 + mr_input_l2) + latency_l2_cache;
    }

    float ratio = 200.f / (Max(cycles_mma, cycles_mem) * std::ceil(cta_launch_times));

    return  ratio;
}


#endif
