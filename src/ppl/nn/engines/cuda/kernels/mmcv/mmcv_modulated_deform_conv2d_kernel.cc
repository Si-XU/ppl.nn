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

#include "ppl/nn/engines/cuda/kernels/mmcv/mmcv_modulated_deform_conv2d_kernel.h"

#include "cudakernel/nn/deform_conv.h"
#include <memory>
#include <fstream>
#include <cuda_fp16.h>

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode MMCVModulatedDeformConv2dKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    auto offset = ctx->GetInput<TensorImpl>(1);
    auto mask = ctx->GetInput<TensorImpl>(2);
    auto weight = ctx->GetInput<TensorImpl>(3);
    auto bias = ctx->GetInputCount() > 4 ? ctx->GetInput<TensorImpl>(4) : nullptr;
    
    auto shape_in0 = ctx->GetInput<TensorImpl>(0)->GetShape();
    auto shape_in3 = ctx->GetInput<TensorImpl>(3)->GetShape();
    auto shape_out = ctx->GetOutput<TensorImpl>(0)->GetShape();

    int64_t size = PPLCUDADeformConvGetBufSize(&shape_in0, &shape_in3, &shape_out);
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
    
    const int64_t num_output = weight->GetShape().GetDim(0);
    const int64_t channels = weight->GetShape().GetDim(1) * param_->groups;
    const int64_t kernel_h = weight->GetShape().GetDim(2);
    const int64_t kernel_w = weight->GetShape().GetDim(3);

    auto stream = GetStream();
    status = PPLCUDADeformConvForward(
        stream, &output->GetShape(), &input->GetShape(), 
        output->GetBufferPtr(), input->GetBufferPtr(), weight->GetBufferPtr(),
        offset->GetBufferPtr(), mask ? mask->GetBufferPtr() : nullptr, bias ? bias->GetBufferPtr() : nullptr,
        param_->groups, param_->deform_groups, channels, num_output,
        param_->stride[0], param_->stride[1], kernel_h, kernel_w, 
        param_->padding[0], param_->padding[1], param_->dilation[0], param_->dilation[1],
        mask, tmp_buffer_desc.addr);
if(strcmp(output->GetEdge()->GetName().c_str(), "442")==0)
{
auto sz = num_output*channels*kernel_h*kernel_w/2;
__half* t = (__half*)malloc(sz*sizeof(__half));
cudaMemcpy(t, (__half*)weight->GetBufferPtr() + sz, sz*sizeof(__half), cudaMemcpyDeviceToHost);
printf("runtime weights: %d\n", sz);
for(int i = 0; i < sz; i++){
    printf("%f\t", (float)t[i]);
}
printf("\n");
free(t);
}
printf("weight shape: %d,%d,%d,%d", num_output,channels,kernel_h,kernel_w);

   
{
    int save_bytes = output->GetShape().GetDataType() == ppl::common::DATATYPE_FLOAT16 ?
                    sizeof(float) * output->GetShape().GetElementsIncludingPadding() :
                    output->GetShape().GetBytesIncludingPadding();
    std::unique_ptr<char[]> out_data(new char[save_bytes]);
    ppl::nn::TensorShape tmp_shape(output->GetShape());
    if(output->GetShape().GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        tmp_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
    } else {
        tmp_shape.SetDataType(output->GetShape().GetDataType());
    }
    tmp_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    output->ConvertToHost(out_data.get(), tmp_shape);
    char t[128];
    strcpy(t, output->GetName());
    strcat(t, ".dat");
    printf("number: %s, %f, %d\n", t, (out_data.get())[0], save_bytes);
    std::ofstream out_fs(t);
    out_fs.write((char*)out_data.get(), save_bytes);
}
    return status;

}

}}} // namespace ppl::nn::cuda
