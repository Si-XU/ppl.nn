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

#include "ppl/nn/engines/cuda/kernels/onnx/cumsum_kernel.h"

#include "cudakernel/nn/cumsum.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode CumSumKernel::DoExecute(KernelExecContext* ctx) {
    // TODO : ADD CumSum Forward
    auto input = ctx->GetInput<TensorImpl>(0);
    auto axis_tensor = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);
    int axis = axis_tensor->GetBufferPtr<int>()[0];


    ppl::common::RetCode status = PPLCUDACumsumForwardImp(GetStream(), axis, input->GetShape(), input->GetBufferPtr(),
                                                            output->GetBufferPtr());
    
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda