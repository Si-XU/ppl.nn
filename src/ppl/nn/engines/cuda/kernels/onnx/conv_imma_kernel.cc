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
        GetCudaDevice()->FreeTmpBuffer(buffer);
    });
    auto d_weight_scale = weight_scale_desc.addr;
    //cudaMemcpy(d_weight_scale, h_weight_scale, qw_size*sizeof(float), cudaMemcpyHostToDevice);
//FIXME
{
float *st = (float*)malloc(qw_size*sizeof(float));
auto sz_per_grp = shape_in1.GetDim(0) / param_->param.group;
auto sz_per_grp_pad = (shape_in1.GetDim(0) / param_->param.group + 15) / 16 * 16;
//printf("st scale:\n");
for(int i = 0; i < qw_size; i++){
    auto g_id = i / sz_per_grp_pad;
    auto id = g_id*sz_per_grp + (i % sz_per_grp_pad);
    st[i] = (i % sz_per_grp_pad) < sz_per_grp? h_weight_scale[id] : 0.f;
    //printf("(%4d,%4f)", i, st[i]);
    //if(i%16==15) printf("\n");
}
    cudaMemcpy(d_weight_scale, st, qw_size*sizeof(float), cudaMemcpyHostToDevice);
free(st);
}

    temp_quant_param.in_scale     = input_scale;
    temp_quant_param.out_scale    = 1 / output_scale;
    temp_quant_param.d_flt_scale  = d_weight_scale;
#if 0
if(strcmp(output->GetName(), "343")==0){
float *t = (float*)malloc(2*56*56*32*sizeof(float));
//cudaMemcpy(t, d_weight_scale, 96*16*sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(t, temp_quant_param.d_flt_scale, 24*16*sizeof(float), cudaMemcpyDeviceToHost);
printf("weight_scale: %x\n", temp_quant_param.d_flt_scale);
for(int i = 0; i < 96; i++){
    printf("%f\t", t[i*16]);
}
printf("weight_scale done\n");
printf("kid:%d, %f, %f\n", param_->extra_param.algo_info.kernel_index, input_scale, output_scale);
cudaMemcpy(t, ctx->GetInput<TensorImpl>(2)->GetBufferPtr(), 24*16*sizeof(float), cudaMemcpyDeviceToHost);
printf("bias:\n");
for(int i = 0; i < 24*16; i++){
    t[i] = 0.f;
    printf("(%4d,%4f)\t", i, t[i]);
if(i%16==15) printf("\n");
}
//cudaMemcpy(ctx->GetInput<TensorImpl>(2)->GetBufferPtr(), t, 24*16*sizeof(float), cudaMemcpyHostToDevice);
printf("bias done\n");

printf("weights:\n");
cudaMemcpy(t, weight->GetBufferPtr(), 24*3*3*16*sizeof(int8_t), cudaMemcpyDeviceToHost);
for(int i = 0; i < 24*3*3*16; i++){
    //if(i%16==0 && i/16/9==0)    printf("(%4d,%4d)\t", i, (int)((int8_t*)t)[i]);
    //if(i%16==0)    ((int8_t*)t)[i] = (int8_t)1;//(i/16%9-4);
    //if(i%16==0)    ((int8_t*)t)[i] = (int8_t)(i/16%9-4);
    printf("(%4d,%4d)\t", i, (int)((int8_t*)t)[i]);
    //if(i/(3*3*16) != 2) ((int8_t*)t)[i] = (int8_t)(rand()%10-5);
if(i%16==15) printf("\n");
    //if(i!=1)    t[i] = 0.f;//t[18];
    //else t[1] = 0.000619;
    //t[i] = t[0];
    //float m = t[63-i];
    //t[63-i] = t[i];
    //t[i] = m;
}
printf("weights done\n");
//cudaMemcpy(weight->GetBufferPtr(), t, 24*3*3*16*sizeof(int8_t), cudaMemcpyHostToDevice);
cudaMemcpy(t, temp_quant_param.d_flt_scale, 24*16*sizeof(float), cudaMemcpyDeviceToHost);
for(int i = 0; i < 24*16; i++){
    //t[i] = 0.f;
    if(i%16==0)    t[i] = 1.f;
    printf("(%4d,%4f)\t", i, t[i]);
if(i%16==15) printf("\n");
}
//cudaMemcpy(temp_quant_param.d_flt_scale, t, 24*16*sizeof(float), cudaMemcpyHostToDevice);
//temp_quant_param.in_scale = 0.01f;
//temp_quant_param.out_scale = 1.f;

free(t);
}
if(strcmp(output->GetName(), "332")==0){
 int a = 1;
}
#endif

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
//{
//if(strcmp(output->GetName(), "332")==0){
//float *t = (float*)malloc(96*16*sizeof(float));
//cudaMemcpy(t, temp_quant_param.d_flt_scale, 96*16*sizeof(float), cudaMemcpyDeviceToHost);
//printf("before conv quant: %d\n", qw_size);
//for(int i = 0; i < 96*16; i++){
//    printf("(%4d,%4f,%4f)\t", i, t[i], h_weight_scale[i/16]);
//if(i%16==15) printf("\n");
//}
//printf("before conv quant done\n");
//free(t);
//}
//}

    PPLCUDAConvolutionForwardImpInt8(
        stream, shape_in0.GetDataType(), (int4*)input->GetBufferPtr(),
        (int4*)weight->GetBufferPtr(), (int4*)output->GetBufferPtr(),
        param_->param.bias_term ? (int4*)ctx->GetInput<TensorImpl>(2)->GetBufferPtr() : nullptr, (int4*)tmp_buffer,
        algo_param, temp_conv_param, temp_quant_param, temp_fuse_param);

    LOG(DEBUG) << "Excute IMMA conv with kernel id:" << param_->extra_param.algo_info.kernel_index
               << " and temp buffer size: " << size;
 
#if 0
//if(shape_in1.GetDim(0)==24&&shape_in1.GetDim(1)==1&&shape_in1.GetDim(2)==3)
if(strcmp(output->GetName(), "343")==0)
{
int8_t *t = (int8_t*)malloc(65*sizeof(int8_t));
cudaMemcpy(t, output->GetBufferPtr(), 65*sizeof(int8_t), cudaMemcpyDeviceToHost);
printf("convint8 output\n");
for(int i = 0; i < 65; i++)
printf("%d\t", (int)((int8_t*)t)[i]);
printf("convint8 output end\n");
}
#endif
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
