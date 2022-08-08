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

#include "ppl/nn/engines/cuda/optimizer/ops/pmx/horizconv_op.h"

#include "ppl/nn/engines/cuda/kernels/pmx/horizconv_hmma_kernel.h"
#include "ppl/nn/engines/cuda/kernels/pmx/horizconv_imma_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_conv.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace cuda {
HorizConvOp::~HorizConvOp() {
    for (uint32_t i = 0; i < param_.extra_param.size; ++i) {
        ConvFusionInfo* fuse_info = &param_.extra_param.fuse_info_list[0];
        for (uint32_t j = 0; j < fuse_info->fuse_attrs.size(); ++j) {
            free(fuse_info->fuse_attrs[i]);
        }
    }
}

void HorizConvOp::CopyParam(void*& param) {
    if (param == nullptr) {
        param = new CudaHorizConvParam();
    }
    *(CudaHorizConvParam*)param = param_;
    return;
}

RetCode HorizConvOp::Init(const OptKernelOptions& options) {

    infer_type_func_ = [this](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        if (type == DATATYPE_INT8) {
            auto in_edge_id = info->GetInput<TensorImpl>(0)->GetEdge()->GetId();
            auto& in_quant = quant->at(in_edge_id);
            auto out_edge_id = info->GetOutput<TensorImpl>(0)->GetEdge()->GetId();
            auto& out_quant = quant->at(out_edge_id);
            if (in_quant.type != DATATYPE_INT8 || out_quant.type != DATATYPE_INT8) {
                return RC_INVALID_VALUE;
            }
            info->GetInput<TensorImpl>(0)->GetShape()->SetDataType(in_quant.type);
            info->GetOutput<TensorImpl>(0)->GetShape()->SetDataType(out_quant.type);

            // Copy quant info skipping input0
            for (uint32_t i = 1; i < info->GetInputCount(); ++i) {
                auto in_edge_id = info->GetInput<TensorImpl>(i)->GetEdge()->GetId();
                auto& in_quant = quant->at(in_edge_id);
                auto in_shape = info->GetInput<TensorImpl>(i)->GetShape();
                if (i == 1 && in_quant.type != DATATYPE_UNKNOWN) {
                    in_shape->SetDataType(in_quant.type);
                    continue;
                }
                in_shape->SetDataType(out_quant.type);
            }

            // Reset bias input type to float32
            for (uint32_t item = 0; item < param_.extra_param.size; ++item) {
                auto weight_index = param_.extra_param.weight_index_list[item];
                auto in_shape = info->GetInput<TensorImpl>(weight_index + 1)->GetShape();
                if (param_.extra_param.bias_term_list[item]) {
                    in_shape->SetDataType(ppl::common::DATATYPE_FLOAT32);
                }
            }
            return ppl::common::RC_SUCCESS;
        }
        type = ppl::common::DATATYPE_FLOAT16;
        return InferDefaultType(info, type);
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto inshape = info->GetInput<TensorImpl>(0)->GetShape();
        if (inshape->GetDimCount() != 4) {
            LOG(DEBUG) << "error input shape dims " << inshape->GetDimCount();
            return ppl::common::RC_INVALID_VALUE;
        }

        for (uint32_t item = 0; item < param_.extra_param.size; ++item) {
            auto weight_index = param_.extra_param.weight_index_list[item];
            auto x = info->GetInput<TensorImpl>(0)->GetShape();
            auto w = info->GetInput<TensorImpl>(weight_index)->GetShape();
            auto y = info->GetOutput<TensorImpl>(item)->GetShape();
            auto num_output = w->GetDim(0);

            y->SetDimCount(x->GetDimCount());
            y->SetDim(0, x->GetDim(0));
            y->SetDim(1, num_output);

            const int32_t kernel_dims = (int32_t)x->GetDimCount() - 2;
            for (int32_t i = 0; i < kernel_dims; ++i) {
                const int32_t j = i + 2;
                const int32_t kernel_shape_eff = (w->GetDim(j) - 1) * param_.param.dilations[i] + 1;
                const int64_t out_dim =
                    (x->GetDim(j) + param_.param.pads[i] + param_.param.pads[i + kernel_dims] - kernel_shape_eff) / param_.param.strides[i] + 1;
                if (out_dim <= 0) {
                    LOG(DEBUG) << "ERROR: output dim[" << out_dim << "] < 0.";
                    return RC_INVALID_VALUE;
                }
                y->SetDim(j, out_dim);
            }
            y->CalcPadding();
        }
        return ppl::common::RC_SUCCESS;
    };

    return RC_SUCCESS;
}

RetCode HorizConvOp::Finalize(const OptKernelOptions& options) {
    param_ = *((CudaHorizConvParam*)options.param);

    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* HorizConvOp::CreateKernelImpl() const {
    if (param_.extra_param.algo_info.algo_type == "HorizTuringHMMAImpgemm") {
        return CreateKernelImplWithParam<HorizConvHmmaKernel>(&param_);
    } else if (param_.extra_param.algo_info.algo_type == "HorizTuringIMMAImpgemm") {
        return CreateKernelImplWithParam<HorizConvImmaKernel>(&param_);
    }
    return nullptr;
}

}}} // namespace ppl::nn::cuda
