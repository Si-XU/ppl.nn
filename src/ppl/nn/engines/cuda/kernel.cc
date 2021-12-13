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

#include "ppl/nn/engines/cuda/kernel.h"

#include <chrono>
#include <memory>
#include <fstream>
#include "ppl/common/allocator.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

CudaKernel::~CudaKernel() {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    if (exec_begin_event_) {
        cudaEventDestroy(exec_begin_event_);
    }
    if (exec_end_event_) {
        cudaEventDestroy(exec_end_event_);
    }
#endif
}

RetCode CudaKernel::Init() {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    auto err = cudaEventCreate(&exec_begin_event_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaEventCreate failed: " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }

    err = cudaEventCreate(&exec_end_event_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaEventCreate failed: " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
#endif
    auto status = barrier_.Init();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "create barrier for kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode CudaKernel::BeforeExecute(KernelExecContext* ctx) {
    auto status = Reshape(ctx);
    if (status != RC_SUCCESS) {
        return status;
    }

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        status = tensor->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << tensor->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
        tensor->SetBarrier(&barrier_);
    }

    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
class CudaTimingGuard final {
public:
    CudaTimingGuard(cudaStream_t stream, cudaEvent_t* begin_event, cudaEvent_t* end_event, bool is_profiling_enabled)
        : is_profiling_enabled_(is_profiling_enabled), end_event_(end_event), stream_(stream) {
        if (is_profiling_enabled) {
            cudaEventRecord(*begin_event, stream);
            stream_ = stream;
        }
    }
    ~CudaTimingGuard() {
        if (is_profiling_enabled_) {
            cudaEventRecord(*end_event_, stream_);
        }
    }

private:
    bool is_profiling_enabled_;
    cudaEvent_t* end_event_;
    cudaStream_t stream_;
};
#endif

bool CudaKernel::CanDoExecute(const KernelExecContext& ctx) const {
    for (uint32_t i = 0; i < ctx.GetInputCount(); ++i) {
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor || tensor->GetShape().GetBytesIncludingPadding() == 0) {
            return false;
        }
    }
    return true;
}

RetCode CudaKernel::Execute(KernelExecContext* ctx) {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    CudaTimingGuard __timing_guard__(static_cast<CudaDevice*>(GetDevice())->GetStream(), &exec_begin_event_,
                                     &exec_end_event_, ctx->IsProfilingEnabled());
#endif

    auto status = BeforeExecute(ctx);
    if (status != RC_SUCCESS) {
        return status;
    }

#ifndef NDEBUG
    uint32_t total_size = 0;
    LOG(INFO) << "Before execute kernel[" << GetName() << "]";
    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
        auto tensor = ctx->GetInput<TensorImpl>(i);
        if (!tensor) {
            continue;
        }
        auto tensor_size = tensor->GetShape().GetBytesIncludingPadding();
        total_size += tensor_size;
    }
    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        auto tensor_size = tensor->GetShape().GetBytesIncludingPadding();
        LOG(DEBUG) << "tensor size " << tensor_size;
        LOG(DEBUG) << "tensor datatype " << tensor->GetShape().GetDataType() << " tensor dataformat "
                   << tensor->GetShape().GetDataFormat();
        LOG(DEBUG) << "tensor dimcount " << tensor->GetShape().GetDimCount();
        LOG(DEBUG) << "tensor n and c " << tensor->GetShape().GetDim(0) << " " << tensor->GetShape().GetDim(1);
        LOG(DEBUG) << "tensor h and w " << tensor->GetShape().GetDim(2) << " " << tensor->GetShape().GetDim(3);
        total_size += tensor_size;
    }
    auto run_begin_ts = std::chrono::system_clock::now();
#endif

    if (CanDoExecute(*ctx)) {
        status = DoExecute(ctx);
    }

// for (uint32_t i = 0; i < ctx->GetOutputCount(); i++) {
//     auto output = ctx->GetOutput<TensorImpl>(i);
//     int save_bytes = output->GetShape().GetDataType() == ppl::common::DATATYPE_FLOAT16 ||
//                         output->GetShape().GetDataType() == ppl::common::DATATYPE_INT8 ?
//                     sizeof(float) * output->GetShape().GetElementsExcludingPadding() :
//                     output->GetShape().GetBytesExcludingPadding();
//     std::unique_ptr<char[]> out_data(new char[save_bytes]);
//     // memset(out_data.get(), 0, save_bytes);
//     TensorShape tmp_shape(output->GetShape());
//     if(output->GetShape().GetDataType() == ppl::common::DATATYPE_FLOAT16) {
//         tmp_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
//     } else if(output->GetShape().GetDataType() == ppl::common::DATATYPE_INT8) {
//         tmp_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
//     } else {
//         tmp_shape.SetDataType(output->GetShape().GetDataType());
//     }
//     tmp_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
//     auto output_quant = common_param_->cuda_tensor_info->at(output->GetEdge()->GetId());
//     CudaTensorQuant convert_quant;
//     ((CudaDataConverter*)GetDevice()->GetDataConverter())->ConvertToHost((void*)out_data.get(), tmp_shape, convert_quant, output->GetBufferDesc(), output->GetShape(), output_quant);
//     std::string outputname = output->GetName();
//     std::ofstream out_fs(outputname + ".dat");
//     out_fs.write((char*)out_data.get(), save_bytes);
// }
#ifndef NDEBUG

    auto run_end_ts = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(run_end_ts - run_begin_ts);
    LOG(INFO) << "After execute kernel[" << GetName() << "] with running time " << (float)diff.count()
              << " ms and memory cost " << total_size;
#endif

    barrier_.Update(static_cast<CudaDevice*>(GetDevice())->GetStream());

    return status;
}

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
uint64_t CudaKernel::GetExecutionTime() const {
    cudaEventSynchronize(exec_end_event_);
    float ms = 0.0;
    cudaEventElapsedTime(&ms, exec_begin_event_, exec_end_event_);
    return static_cast<uint64_t>(ms * 1000);
}
#endif

}}} // namespace ppl::nn::cuda
