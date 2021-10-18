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

#include "ppl/nn/engines/cuda/optimizer/algos/algo_deform_conv.h"

#include <chrono>

#include "cudakernel/nn/deform_conv.h"
#include "ppl/common/cuda/cuda_types.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/utils.h"
#include <cuda_fp16.h>

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

double DeformConvAlgorithm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    return 1e-5;
}

RetCode DeformConvAlgorithm::ModifyParam(const ir::Node* node, OptKernelOptions& options) {
    this->param_ = (reinterpret_cast<ppl::nn::common::MMCVModulatedDeformConv2dParam*>(options.param));
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto weight_edge = topo->GetEdgeById(node->GetInput(3));
    auto weight_node = topo->GetNodeById(weight_edge->GetProducer());

    auto shape_in1 = options.tensors->find(node->GetInput(3))->second->GetShape();

    RetCode status;
    
    // Split weight format to group padding
    auto stream = options.device->GetStream();
    auto weight_iter = data->constants.find(weight_node->GetInput(0));
    if (weight_iter != data->constants.end() && // is a constant tensor and has not be loaded
        options.info->constants.find(weight_node->GetInput(0)) == options.info->constants.end()) {
        auto preedge_id = weight_node->GetInput(0);
        auto postedge_id = node->GetInput(3);
        auto preshape = options.tensors->find(preedge_id)->second->GetShape();
        auto postshape = options.tensors->find(postedge_id)->second->GetShape();
        auto size = postshape.GetElementsIncludingPadding();
        size = (size / postshape.GetDim(0) + 7)/ 8 * 8 * postshape.GetDim(0);

        RuntimeConstantInfo weight_constat_info;
printf("in modify out weight 1\n");
        {
            BufferDesc buffer;
            status = options.device->Realloc(size, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            weight_constat_info.Reshape(postshape); // give the init shape, but the actual shape is padded
            weight_constat_info.SetBuffer(buffer, options.device, true);
        }

        ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, postshape.GetBytesIncludingPadding(), RC_OUT_OF_MEMORY)
//if(strcmp(topo->GetEdgeById(weight_node->GetInput(0))->GetName().c_str(), "backbone.layer2.0.conv2.weight") == 0)
//{
//int64_t sz = size;
//float *t = (float*)malloc(sz*sizeof(float));
//cudaMemcpy(t, weight_iter->second.data.data(), sz*sizeof(float), cudaMemcpyHostToHost);
//for(int i = 0; i < sz; i++)
//    printf("%f\t", (float)t[i]);
//free(t);
//printf("\nin modify weight iter:\n");
//}
        status = options.device->GetDataConverter()->ConvertFromHost(&temp_buffer, postshape,
                                                                     weight_iter->second.data.data(), preshape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << node->GetName() << " copy constant failed: " << GetRetCodeStr(status);
            return status;
        }
//if(strcmp(topo->GetEdgeById(weight_node->GetInput(0))->GetName().c_str(), "backbone.layer2.0.conv2.weight") == 0)
//{
//printf("in modify out weight\n");
//int64_t sz = size;
//__half *t = (__half*)malloc(sz*sizeof(__half));
//cudaMemcpy(t, temp_buffer.addr, sz*sizeof(__half), cudaMemcpyDeviceToHost);
//for(int i = 0; i < sz; i++)
//    printf("%f\t", (float)t[i]);
//free(t);
//}
//cudaMemcpy(weight_constat_info.GetBufferDesc().addr, temp_buffer.addr, size*sizeof(__half), cudaMemcpyDeviceToDevice);
        PPLCUDADeformConvModifyWeights(stream, &postshape, temp_buffer.addr, weight_constat_info.GetBufferDesc().addr);
//{
//printf("in modify out weight 2\n");
//int64_t sz = size;
//__half *t = (__half*)malloc(sz*sizeof(__half));
//cudaMemcpy(t, weight_constat_info.GetBufferDesc().addr, sz*sizeof(__half), cudaMemcpyDeviceToHost);
//for(int i = 0; i < sz; i++)
//    printf("%f\t", (float)t[i]);
//free(t);
//}
if(strcmp(topo->GetEdgeById(weight_node->GetInput(0))->GetName().c_str(), "backbone.layer2.0.conv2.weight") == 0)
{
int sz = size/2;
printf("origin weight: %d\n", sz);
__half *t = (__half*)malloc(sz*sizeof(__half));
cudaMemcpy(t, (__half*)weight_constat_info.GetBufferDesc().addr+ size/2, sz*sizeof(__half), cudaMemcpyDeviceToHost);
for(int i = 0; i < sz; i++)
    printf("%f\t", (float)t[i]);
printf("\n");
}


        options.info->constants.emplace(preedge_id, std::move(weight_constat_info));
        options.tensors->find(preedge_id)->second->GetShape() = postshape;
        options.quants->at(preedge_id).format = postshape.GetDataFormat();
        options.quants->at(preedge_id).type = postshape.GetDataType();
    }

    return RC_SUCCESS;
}

void DeformConvAlgorithm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                       dataformat_t input_format, dataformat_t output_format) {
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) { // only reset formats of input0 and weight
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto shape = &tensors->find(edge_id)->second->GetShape();
        if (shape->GetDimCount() > 1)
            shape->SetDataFormat(input_format);
        else
            shape->SetDataFormat(DATAFORMAT_NDARRAY);
    }

    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto shape = &tensors->find(edge_id)->second->GetShape();
        shape->SetDataFormat(output_format);
    }
    return;
}

}}} // namespace ppl::nn::cuda
