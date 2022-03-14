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

#include "ppl/nn/engines/cuda/kernels/onnx/matmul_fp32_kernel.h"

#include "cudakernel/matmul.h"

#include "ppl/nn/common/logger.h"
namespace ppl { namespace nn { namespace cuda {

bool MatMulFp32Kernel::CanDoExecute(const KernelExecContext& ctx) const {
    const TensorShape& input0 = *ctx.GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& input1 = *ctx.GetInput<TensorImpl>(1)->GetShape();
    if (input0.GetBytesIncludingPadding() == 0) {
        return false;
    }
    if (input1.GetBytesIncludingPadding() == 0) {
        return false;
    }
    // K must be the same
    uint32_t size0 = input0.GetDimCount();
    uint32_t size1 = input1.GetDimCount();
    if (input0.GetDim(size0 - 1) != input1.GetDim(size1 - 2)) {
        return false;
    }

    return true;
}

ppl::common::RetCode MatMulFp32Kernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);
    // LOG(ERROR) << "run matmul";
    // LOG(ERROR) << input0 << " " << weight << " " << output;

    // auto in_size = input0->GetShape()->GetElementsExcludingPadding();
    // auto wei_size = weight->GetShape()->GetElementsExcludingPadding();
    // auto out_size = output->GetShape()->GetElementsExcludingPadding();
    // float input_t[in_size];
    // float weight_t[wei_size];
    // float output_t[out_size];

    // LOG(ERROR) << input0->GetBufferPtr();
    // LOG(ERROR) << in_size;

    // input0->CopyToHost(input_t);
    // weight->CopyToHost(weight_t);

    // auto dim_count = input0->GetShape()->GetDimCount();
    // auto M = input0->GetShape()->GetDim(dim_count - 2);
	// auto K = input0->GetShape()->GetDim(dim_count - 1);
	// auto N = output->GetShape()->GetDim(dim_count - 1);

    // float sum = 0.0f;
    // for (uint32_t k = 0; k < K; k++) {
    //     sum += input_t[k] * weight_t[k * N];
    // }
    // LOG(ERROR) << "sum for 0 index " << sum;

    auto status = PPLCUDAMatmulForwardImp(GetStream(), input0->GetShape(), input0->GetBufferPtr(),
        weight->GetShape(), weight->GetBufferPtr(), output->GetShape(), output->GetBufferPtr());
    
    return status;
}

}}} // namespace ppl::nn::cuda
