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

#include "ppl/nn/engines/cuda/kernels/onnx/conv_imma_kernel.h"

#include <cuda_fp16.h>

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode ConvImmaKernel::BeforeExecute(KernelExecContext* ctx) {
    auto status = Reshape(ctx);
    if (status != ppl::common::RC_SUCCESS) {
        return status;
    }

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        auto device = GetCudaDevice();
        auto concat_edge_id = param_->extra_param.fuse_info.concat_edge_id;
        if (param_->extra_param.fuse_info.channel_offset >= 0) {
            auto edge2buffer = device->GetEdge2Buffer();
            auto ptr = edge2buffer->find(concat_edge_id);
            if (ptr == edge2buffer->end()) {
                BufferDesc buffer;
                auto concat_shape = tensor->GetShape();
                concat_shape.SetDim(1, param_->extra_param.fuse_info.channel_size);
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

ppl::common::RetCode ConvImmaKernel::DoExecute(KernelExecContext* ctx) {
    conv_param_t temp_conv_param;
    fuse_param_t temp_fuse_param;
    quant_param_t temp_quant_param;

    auto input = ctx->GetInput<TensorImpl>(0);
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);
    auto shape_in0 = input->GetShape();
    auto shape_in1 = weight->GetShape();
    auto shape_out = output->GetShape();

    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input->GetEdge()->GetId());
    auto weight_quant = GetCommonParam()->cuda_tensor_info->at(weight->GetEdge()->GetId());
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output->GetEdge()->GetId());
    auto input_scale = input_quant.scale[0];
    int qw_size = (shape_in1.GetDim(0) / param_->param.group + 15) / 16*16;
    if (!weight_quant.per_chnnal) {
        weight_quant.scale.insert(weight_quant.scale.begin(), qw_size, weight_quant.scale[0]);
    }
    auto h_weight_scale = weight_quant.scale.data();
    auto output_scale = output_quant.scale[0];

    BufferDesc weight_scale_desc;
    GetCudaDevice()->AllocTmpBuffer(qw_size*sizeof(float), &weight_scale_desc);
    BufferDescGuard __weight_scale_guard(&weight_scale_desc, [this](BufferDesc* buffer) -> void {
        GetCudaDevice()->FreeTmpBuffer(buffer);
    });
    auto d_weight_scale = weight_scale_desc.addr;
    cudaMemcpy(d_weight_scale, h_weight_scale, qw_size*sizeof(float), cudaMemcpyHostToDevice);

    temp_quant_param.in_scale     = input_scale;
    temp_quant_param.out_scale    = 1 / output_scale;
    temp_quant_param.d_flt_scale  = d_weight_scale;
//{
//float *t = (float*)malloc(64*64*3*3*sizeof(float));
////cudaMemcpy(t, d_weight_scale, 64*sizeof(float), cudaMemcpyDeviceToHost);
//printf("kid:%d, %f, %f\n", param_->extra_param.algo_info.kernel_index, input_scale, output_scale);
////for(int i = 0; i < 64; i++){
////    printf("%d: %f\t", i, t[i]);
//cudaMemcpy(t, weight->GetBufferPtr(), 64*64*3*3*sizeof(int8_t), cudaMemcpyDeviceToHost);
//for(int i = 0; i < 64*64*3*3; i++){
//    printf("%d: %d\t", i, (int)((int8_t*)t)[i]);
//    //if(i!=1)    t[i] = 0.f;//t[18];
//    //else t[1] = 0.000619;
//    //t[i] = t[0];
//    //float m = t[63-i];
//    //t[63-i] = t[i];
//    //t[i] = m;
//}
////cudaMemcpy(d_weight_scale, t, 64*sizeof(float), cudaMemcpyHostToDevice);
////cudaMemcpy(ctx->GetInput<TensorImpl>(2)->GetBufferPtr(), t, 64*sizeof(float), cudaMemcpyHostToDevice);
////    t[1] = 1.f;//t[18];
////cudaMemcpy(d_weight_scale, t, 64*sizeof(float), cudaMemcpyHostToDevice);
//
//free(t);
//}

    ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, param_->param, temp_conv_param);
    ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info, temp_fuse_param);

    struct algo_param_t algo_param;
    algo_param.kid = param_->extra_param.algo_info.kernel_index;
    algo_param.splitk = param_->extra_param.algo_info.splitk;
    algo_param.splitf = param_->extra_param.algo_info.splitf;

    uint64_t size = PPLCUDAConvolutionGetRuntimeBufSize(shape_in0.GetDataType(), temp_conv_param, algo_param.splitk,
                                                        algo_param.splitf, ((uint64_t)8) * 1024 * 1024 * 1024);

    BufferDesc tmp_buffer_desc;
    auto status = GetCudaDevice()->AllocTmpBuffer(size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    BufferDescGuard __tmp_buffer_guard(&tmp_buffer_desc, [this](BufferDesc* buffer) -> void {
        GetCudaDevice()->FreeTmpBuffer(buffer);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;

    auto stream = GetStream();
    PPLCUDAConvolutionForwardImpInt8(
        stream, shape_in0.GetDataType(), (int4*)input->GetBufferPtr(),
        (int4*)weight->GetBufferPtr(), (int4*)output->GetBufferPtr(),
        param_->param.bias_term ? (int4*)ctx->GetInput<TensorImpl>(2)->GetBufferPtr() : nullptr, (int4*)tmp_buffer,
        algo_param, temp_conv_param, temp_quant_param, temp_fuse_param);

    LOG(DEBUG) << "Excute IMMA conv with kernel id:" << param_->extra_param.algo_info.kernel_index
               << " and temp buffer size: " << size;
 
//{
//int8_t *t = (int8_t*)malloc(65*sizeof(int8_t));
//cudaMemcpy(t, output->GetBufferPtr(), 65*sizeof(int8_t), cudaMemcpyDeviceToHost);
//printf("convint8 output\n");
//for(int i = 0; i < 65; i++)
//printf("%d\t", (int)((int8_t*)t)[i]);
//printf("convint8 output end\n");
//}
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
