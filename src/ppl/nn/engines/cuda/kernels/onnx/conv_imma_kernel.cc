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
    int qw_size = ((shape_in1.GetDim(0) / param_->param.group + 15) / 16*16) * param_->param.group;
    if (!weight_quant.per_chnnal) {
        weight_quant.scale.insert(weight_quant.scale.begin(), qw_size, weight_quant.scale[0]);
    }
    auto h_weight_scale = weight_quant.scale.data();
    auto output_scale = output_quant.scale[0];

    BufferDesc weight_scale_desc;
    GetCudaDevice()->Realloc(qw_size*sizeof(float), &weight_scale_desc);
    BufferDescGuard __weight_scale_guard(&weight_scale_desc, [this](BufferDesc* buffer) -> void {
        GetCudaDevice()->Free(buffer);
    });
    auto d_weight_scale = weight_scale_desc.addr;
//FIXME
{
float *st = (float*)malloc(qw_size*sizeof(float));
auto sz_per_grp = shape_in1.GetDim(0) / param_->param.group;
auto sz_per_grp_pad = (shape_in1.GetDim(0) / param_->param.group + 15) / 16 * 16;
for(int i = 0; i < qw_size; i++){
    auto g_id = i / sz_per_grp_pad;
    auto id = g_id*sz_per_grp + (i % sz_per_grp_pad);
    st[i] = (i % sz_per_grp_pad) < sz_per_grp? h_weight_scale[id] : 0.f;
}
    cudaMemcpy(d_weight_scale, st, qw_size*sizeof(float), cudaMemcpyHostToDevice);
free(st);
}

    temp_quant_param.in_scale     = input_scale;
    temp_quant_param.out_scale    = 1 / output_scale;
    temp_quant_param.d_flt_scale  = d_weight_scale;


    ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, param_->param, temp_conv_param);
    ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info, temp_fuse_param);

    if (temp_fuse_param.has_elt) {
        //auto tps = param_->extra_param.fuse_info.types;
        //auto ret = std::find(tps.begin(), tps.end(), "Add")
        //if (ret == tps.end())
        //    LOG(ERROR) << "fuse_info types error: no add op";
        //int  id  =  ret - tps.begin();
        //auto elt_id = param_extra_param.fuse_info.ind[id];
        //auto elt = ctx->GetInput<TensorImpl>(3);
        //auto elt_quant = GetCommonParam()->cuda_tensor_info->at(elt->GetEdge()->GetId());
        //temp_quant_param.pre_scale = elt_quant.scale[0];
        temp_quant_param.pre_scale = output_scale;
    }
    if (param_->extra_param.fuse_info.channel_offset >= 0) {
         temp_quant_param.out_scale = 1 / GetCommonParam()->cuda_tensor_info->at(param_->extra_param.fuse_info.concat_edge_id).scale[0];
    }


    struct algo_param_t algo_param = param_->extra_param.algo_info;

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

#ifdef PPLNN_ENABLE_CUDA_JIT
    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);
    PPLCUDAConvolutionForwardJitImpInt8(
        stream, module->GetKernelFunc(), shape_in0.GetDataType(), (int4*)input->GetBufferPtr(),
        (int4*)weight->GetBufferPtr(), (int4*)output->GetBufferPtr(),
        param_->param.bias_term ? (int4*)ctx->GetInput<TensorImpl>(2)->GetBufferPtr() : nullptr, (int4*)tmp_buffer,
        algo_param, temp_conv_param, temp_quant_param, temp_fuse_param);    
#else
    PPLCUDAConvolutionForwardImpInt8(
        stream, shape_in0.GetDataType(), (int4*)input->GetBufferPtr(),
        (int4*)weight->GetBufferPtr(), (int4*)output->GetBufferPtr(),
        param_->param.bias_term ? (int4*)ctx->GetInput<TensorImpl>(2)->GetBufferPtr() : nullptr, (int4*)tmp_buffer,
        algo_param, temp_conv_param, temp_quant_param, temp_fuse_param);
#endif
    LOG(DEBUG) << "Excute IMMA conv with kernel id:" << param_->extra_param.algo_info.kid
               << " and temp buffer size: " << size;

    // {
    //     auto output = ctx->GetOutput<TensorImpl>(0);
    //     int8_t* a = new int8_t[1*128*28*28];
    //     output->CopyToHost(a);
    //     for(int i = 0; i < 100; i++) {
    //         printf("%d, %d %f \n", i, a[i * 128], a[i * 128] * output_scale);
    //         // printf("%d, %d %f \n", i, a[i], a[i] * output_scale);
    //     }
    // }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
