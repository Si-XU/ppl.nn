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

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_conv.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"

#define ADDPARAM(horiz_param, conv_param) {                                                                          \
    horiz_param->extra_param.size++;                                                                                 \
    horiz_param->extra_param.fuse_info_list.push_back(conv_param->extra_param.fuse_info);                            \
    horiz_param->extra_param.bias_term_list.push_back(conv_param->extra_param.bias_term);                            \
    horiz_param->extra_param.weight_index_list.push_back(node->GetInputCount());                                     \
    horiz_param->extra_param.is_initializer_weight_list.push_back(conv_param->extra_param.is_initializer_weight);    \
 }

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

const RetCode ConvFusion::FuseConvWithNextNode(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto connect_edge_id = node->GetOutput(0);

    for (uint32_t i = 0; i < nextnode->GetOutputCount(); ++i) {
        auto edge_id = nextnode->GetOutput(i);
        auto temp_edge = topo->GetEdge(edge_id);
        temp_edge->SetProducer(node->GetId());
        if (i == 0) {
            node->ReplaceOutput(connect_edge_id, edge_id);
        } else {
            node->AddOutput(edge_id);
        }
    }

    for (uint32_t i = 0; i < nextnode->GetInputCount(); ++i) {
        auto edge_id = nextnode->GetInput(i);
        if (edge_id == connect_edge_id || edge_id == INVALID_EDGEID) {
            continue;
        }
        ir::Edge* edge = topo->GetEdge(edge_id);
        edge->DelConsumer(nextnode->GetId());
        edge->AddConsumer(node->GetId());
        node->AddInput(edge_id);
    }

    topo->DelEdge(connect_edge_id);
    topo->DelNode(nextnode->GetId());
    return RC_SUCCESS;
}

const RetCode ConvFusion::FuseConvWithBrotherNode(ir::Node* node, ir::Node* brothernode, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto connect_edge_id = node->GetInput(0);
    auto connect_edge = topo->GetEdge(connect_edge_id);

    for (uint32_t i = 0; i < brothernode->GetOutputCount(); ++i) {
        auto edge_id = brothernode->GetOutput(i);
        auto temp_edge = topo->GetEdge(edge_id);
        temp_edge->SetProducer(node->GetId());
        node->AddOutput(edge_id);
    }

    for (uint32_t i = 0; i < brothernode->GetInputCount(); ++i) {
        auto edge_id = brothernode->GetInput(i);
        if (edge_id == connect_edge_id || edge_id == INVALID_EDGEID) {
            continue;
        }
        ir::Edge* edge = topo->GetEdge(edge_id);
        edge->DelConsumer(brothernode->GetId());
        edge->AddConsumer(node->GetId());
        node->AddInput(edge_id);
    }

    connect_edge->DelConsumer(brothernode->GetId());
    topo->DelNode(brothernode->GetId());
    return RC_SUCCESS;
}

const bool ConvFusion::FuseVerti(ir::Node* node, const OptKernelOptions& options,
                                std::function<ppl::common::RetCode(ir::Node*, const OptKernelOptions&)> canfuse) {
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto node_id = node->GetId();
    auto opt_kernel = (CudaOptKernel*)(options.info->kernels[node_id].get());
    CudaConvParam* param = (CudaConvParam*)opt_kernel->GetParam();

    auto edge_id = node->GetOutput(0);
    auto edge = topo->GetEdge(edge_id);
    if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
        return false;
    }
    if (topo->GetEdge(edge_id)->CalcConsumerCount() != 1) { // Can not fuse multi-consumer edge
        return false;
    }

    auto nextnode_id = topo->GetEdge(edge_id)->CreateConsumerIter().Get(); // Get Output(0)
    auto nextnode = topo->GetNode(nextnode_id);

    if (canfuse(nextnode, options)) {
        LOG(DEBUG) << "Fuse node[" << node->GetName() << "] and nextnode[" << nextnode->GetName() << "]";
        // avoid conv+add+add case
        for (auto& type : param->extra_param.fuse_info.types) {
            if (type == nextnode->GetType().name)
                return false;
        }
        param->extra_param.fuse_info.types.emplace_back(nextnode->GetType().name);
        param->extra_param.fuse_info.input_ind.emplace_back(node->GetInputCount());

        if (nextnode->GetType().name != "Clip") {
            auto next_kernel = (CudaOptKernel*)(options.info->kernels[nextnode_id].get());
            void* temp_param = nullptr;
            next_kernel->CopyParam(temp_param);
            param->extra_param.fuse_info.fuse_attrs.emplace_back(std::move(temp_param));
        } else {
            auto clip_param = new CudaClipParam();
            auto min_iter = data->constants.find(nextnode->GetInput(1));
            if (min_iter != data->constants.end()) {
                clip_param->min_value = *(float*)(min_iter->second.data.GetData());
            }
            auto max_iter = data->constants.find(nextnode->GetInput(2));
            if (max_iter != data->constants.end()) {
                clip_param->max_value = *(float*)(max_iter->second.data.GetData());
            }
            param->extra_param.fuse_info.fuse_attrs.emplace_back((void*)clip_param);
        }
        options.info->kernels.erase(nextnode_id);
        FuseConvWithNextNode(node, nextnode, options);
        return true;
    }
    return false;
}

const bool ConvFusion::FuseHoriz(ir::Node* node, bool reliable, const OptKernelOptions& options,
                                std::function<ppl::common::RetCode(ir::Node*, ir::Node*, const OptKernelOptions&)> canfuse) {
    auto topo = options.graph->topo.get();
    auto node_id = node->GetId();
    auto self_opt_kernel = (CudaOptKernel*)(options.info->kernels[node_id].get());
    CudaConvParam* self_param = (CudaConvParam*)self_opt_kernel->GetParam();

    auto edge_id = node->GetInput(0);
    auto edge = topo->GetEdge(edge_id);
    if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
        return false;
    }
    if (topo->GetEdge(edge_id)->CalcConsumerCount() == 1) { // Can not fuse single comsumer edge
        return false;
    }
    
    std::vector<ir::Node*> node_list;
    for (auto iter = topo->GetEdge(edge_id)->CreateConsumerIter(); iter.IsValid(); iter.Forward()) {
        auto brothernode_id = iter.Get();
        auto brothernode = topo->GetNode(brothernode_id);
        if (node_id != brothernode_id && canfuse(node, brothernode, options)) {
            node_list.push_back(brothernode);
        }
    }

    if (node_list.size() > 0) {
        node->SetType(ir::Node::Type("pmx", "HorizConv", 1));
        auto creator = OptKernelCreatorManager::GetInstance()->Find("pmx", "HorizConv", 1);
        if (!creator) {
            LOG(ERROR) << "Cannot find creator for horiz conv kernel";
            return false;
        }

        auto opt_kernel = unique_ptr<CudaOptKernel>((*creator)(node));
        if (!opt_kernel) {
            LOG(ERROR) << "create Kernel failed: oom";
            return false;
        }
        auto param = (CudaHorizConvParam*)(opt_kernel->GetParam());
        if (param == nullptr) {
            LOG(ERROR) << "Can not find param.";
            return false;
        }

        ADDPARAM(param, self_param);
        param->param = self_param->param;
        param->extra_param.algo_info = self_param->extra_param.algo_info;
        param->extra_param.weight_index_list[0] = 1;

        for (uint32_t i = 0; i < node_list.size(); ++i) {
            auto brothernode = node_list[i];
            LOG(DEBUG) << "Fuse node[" << node->GetName() << "] and brothernode[" << brothernode->GetName() << "]";

            // Fuse vertical nodes first
            FuseVerti(brothernode, options, CanFuseRelu);
            if (reliable) {
                if (FuseVerti(brothernode, options, CanFuseElementwise)) {
                    FuseVerti(brothernode, options, CanFuseRelu);
                }
            }

            auto brothernode_id = brothernode->GetId();
            auto brother_opt_kernel = (CudaOptKernel*)(options.info->kernels[brothernode_id].get());
            CudaConvParam* brother_param = (CudaConvParam*)brother_opt_kernel->GetParam();
            ADDPARAM(param, brother_param);
            options.info->kernels.erase(brothernode_id);
            FuseConvWithBrotherNode(node, brothernode, options);
        }

        opt_kernel->Init(options);
        options.info->kernels.erase(node_id);
        options.info->kernels.emplace(node_id, std::move(opt_kernel));
    }
    return false;
}

const RetCode ConvFusion::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    FuseVerti(node, options, CanFuseRelu);
    if (reliable) {
        if (FuseVerti(node, options, CanFuseElementwise)) {
            FuseVerti(node, options, CanFuseRelu);
        }
    }
    FuseHoriz(node, reliable, options, CanFuseConv);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
