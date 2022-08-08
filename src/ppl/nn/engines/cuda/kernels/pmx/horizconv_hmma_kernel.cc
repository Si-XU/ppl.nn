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

#include "ppl/nn/engines/cuda/kernels/pmx/horizconv_hmma_kernel.h"
#include "ppl/common/cuda/cuda_types.h"
#include "ppl/common/destructor.h"
#include <cuda_fp16.h>

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode HorizConvHmmaKernel::BeforeExecute(KernelExecContext* ctx) {
    auto status = Reshape(ctx);
    if (status != ppl::common::RC_SUCCESS) {
        return status;
    }

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        auto device = GetCudaDevice();
        tensor->SetDevice(device);
        auto concat_edge_id = param_->extra_param.fuse_info_list[i].concat_edge_id;
        if (param_->extra_param.fuse_info_list[i].channel_offset >= 0) {
            auto edge2buffer = device->GetEdge2Buffer();
            auto ptr = edge2buffer->find(concat_edge_id);
            if (ptr == edge2buffer->end()) {
                BufferDesc buffer;
                auto concat_shape = *tensor->GetShape();
                auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(concat_shape.GetDataFormat());
                auto channel_size = param_->extra_param.fuse_info_list[i].channel_size;
                auto channel_size_pad = (channel_size + align_size - 1) / align_size * align_size;
                concat_shape.SetDim(1, channel_size_pad);
                status = device->Realloc(concat_shape, &buffer);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                    return status;
                }
                tensor->SetBuffer(buffer);
                edge2buffer->emplace(concat_edge_id, std::move(buffer));
            } else {
                tensor->SetBuffer(ptr->second);
            }
        } else {
            status = tensor->ReallocBuffer();
        }
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << tensor->GetName() << "] failed.";
            return status;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode HorizConvHmmaKernel::DoExecute(KernelExecContext* ctx) {
    conv_param_t temp_conv_param;
    fuse_param_t temp_fuse_param[param_->extra_param.HORIZ_SIZE];
    quant_param_t temp_quant_param[param_->extra_param.HORIZ_SIZE];
    void* weight_ptr[param_->extra_param.HORIZ_SIZE];
    void* output_ptr[param_->extra_param.HORIZ_SIZE];
    void* bias_ptr[param_->extra_param.HORIZ_SIZE];
    int32_t c_out = 0;

    auto input = ctx->GetInput<TensorImpl>(0);
    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input->GetEdge()->GetId());
    auto input_scale = input_quant.scale[0];

    
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);
    TensorShape& shape_in0 = *input->GetShape();
    TensorShape& shape_in1 = *weight->GetShape();
    TensorShape& shape_out = *output->GetShape();

    CudaConvParam temp_attr_param;
    temp_attr_param.param = param_->param;
    temp_attr_param.extra_param.bias_term = param_->extra_param.bias_term_list[0];
    ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, temp_attr_param, temp_conv_param);
    struct algo_param_t algo_param = param_->extra_param.algo_info;

    for (uint32_t item = 0; item < param_->extra_param.size; item++) {
        auto weight_index = param_->extra_param.weight_index_list[item];
        weight = ctx->GetInput<TensorImpl>(weight_index);
        output = ctx->GetOutput<TensorImpl>(item);

        weight_ptr[item] = weight->GetBufferPtr();
        output_ptr[item] = output->GetBufferPtr();
        bias_ptr[item] = nullptr;
        if (param_->extra_param.bias_term_list[item]) {
            bias_ptr[item] = ctx->GetInput<TensorImpl>(weight_index + 1)->GetBufferPtr();
        }
        ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info_list[item], temp_fuse_param[item]);
        c_out += weight->GetShape()->GetDim(0);
    }
    temp_conv_param.num_flt = c_out;

    auto stream = GetStream();
    int device_id = GetDeviceId();

// TODO: complete forward function here
// #ifdef PPLNN_ENABLE_CUDA_JIT
//     CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);
//     PPLCUDAConvolutionForwardJitImp(
//         device_id, stream, module->GetKernelFunc(), shape_in0.GetDataType(), (int4*)ctx->GetInput<TensorImpl>(0)->GetBufferPtr(),
//         (int4**)weight_ptr, (int4**)output_ptr,
//         (int4**)bias_ptr,
//         algo_param, temp_conv_param, temp_fuse_param);
// #else
//     PPLCUDAConvolutionForwardImp(
//         device_id, stream, shape_in0.GetDataType(), (int4*)ctx->GetInput<TensorImpl>(0)->GetBufferPtr(),
//         (int4**)weight_ptr, (int4**)output_ptr,
//         (int4**)bias_ptr,
//         algo_param, temp_conv_param, temp_fuse_param);
// #endif
    LOG(DEBUG) << "Excute HMMA horizconv with kernel id:" << param_->extra_param.algo_info.kid;
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
