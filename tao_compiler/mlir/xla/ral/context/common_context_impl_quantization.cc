// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if defined(TAO_CPU_ONLY) && defined(TAO_AARCH64)

#include <sstream>
#include <thread>

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "tensorflow/compiler/mlir/xla/ral/context/common_context_impl_mkldnn.h"

namespace tao {
namespace ral {

using namespace arm_compute;

namespace {

struct AclConvInfo {
  arm_compute::Tensor src;
  arm_compute::Tensor weights;
  arm_compute::Tensor dst;
  NEConvolutionLayer conv;
};

template <typename TKey, typename TValue>
using ACLQConvMap = std::unordered_map<TKey, TValue, ConvParamsKeyHasher>;
using ACLQConvCache =
    ideep::utils::lru_cache<ConvParamsKey, std::shared_ptr<AclConvInfo>,
                            ACLQConvMap>;

struct ACLQConvState : public Context::Resource {
  std::mutex mu;
  ACLQConvCache cache{getWeightPrePackingCacheCapacity()};
};

template <int NDims>
void ral_qconv_s8_s8_s8(
    ExecutionContext* ctx, opaque_t /*stream_handle*/,
    MemRefType<int8_t, NDims> input, MemRefType<int8_t, NDims> kernel,
    MemRefType<int32_t, 1> padding, MemRefType<float, 0> inputScales,
    MemRefType<float, 1> filterScales, MemRefType<float, 0> outputScales,
    MemRefType<int8_t, NDims> output, MemRefType<int32_t, 1> metadata) {
  CpuTimer timer("ral_qconv_s8_s8_s8");
  if (isEmptyMemref(input) || isEmptyMemref(kernel) || isEmptyMemref(output)) {
    TAO_VLOG(1) << "ral_qconv_s8_s8_s8: early return for empty tensor";
    return;
  }
  ConvParams params;
  if (!parseConvParams(ctx, input, kernel, padding, output, metadata,
                       &params)) {
    ctx->signalError(Context::FAILURE, "invalid conv params");
  }

  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(0) << "input scale = " << inputScales.data[0];
    TAO_VLOG(0) << "output scale = " << outputScales.data[0];
    for (int i = 0; i < filterScales.sizes[0]; ++i)
      TAO_VLOG(0) << "filter_scale[" << i << "] = " << filterScales.data[i];
  }

  if (params.groups > 1) {
    ctx->signalError(Context::FAILURE, "invalid conv params");
  }

  int N = input.sizes[0];
  int H = input.sizes[1];
  int W = input.sizes[2];
  int Ci = input.sizes[3];
  int Co = output.sizes[3];
  int Kh = kernel.sizes[1];
  int Kw = kernel.sizes[2];

  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(1) << "N = " << N;
    TAO_VLOG(1) << "H = " << H;
    TAO_VLOG(1) << "W = " << W;
    TAO_VLOG(1) << "Ci = " << Ci;
    TAO_VLOG(1) << "Co = " << Co;
    TAO_VLOG(1) << "Kh = " << Kh;
    TAO_VLOG(1) << "Kw = " << Kw;
    TAO_VLOG(0) << "params.strides[1] = " << params.strides[1];
    TAO_VLOG(0) << "params.strides[0] = " << params.strides[0];
    TAO_VLOG(0) << "params.padding_l[1] = " << params.padding_l[1];
    TAO_VLOG(0) << "params.padding_l[0] = " << params.padding_l[0];
    TAO_VLOG(0) << "params.padding_r[1] = " << params.padding_r[1];
    TAO_VLOG(0) << "params.padding_r[0] = " << params.padding_r[0];
    TAO_VLOG(0) << "params.dilates[1] = " << params.dilates[1];
    TAO_VLOG(0) << "params.dilates[0] = " << params.dilates[0];
  }

  auto AclConvCreator = [&]() {
    std::shared_ptr<AclConvInfo> info(new AclConvInfo);
    DataLayout data_layout = DataLayout::NHWC;

    TensorShape src_shape, weights_shape, biases_shape, dst_shape;
    src_shape = TensorShape(Ci, W, H, N);
    weights_shape = TensorShape(Ci, Kw, Kh, Co);
    dst_shape = TensorShape(Co, W, H, N);

    DataType data_type = DataType::QASYMM8_SIGNED;
    TensorInfo src_info = TensorInfo(src_shape, 1, data_type, data_layout);
    TensorInfo weights_info =
        TensorInfo(weights_shape, 1, DataType::QSYMM8_PER_CHANNEL, data_layout);
    TensorInfo dst_info = TensorInfo(dst_shape, 1, data_type, data_layout);

    const QuantizationInfo src_qinfo = QuantizationInfo();
    src_info.set_quantization_info(QuantizationInfo(*inputScales.data, 0));
    std::vector<float> scales(filterScales.data,
                              filterScales.data + filterScales.sizes[0]);
    weights_info.set_quantization_info(QuantizationInfo(std::move(scales)));
    dst_info.set_quantization_info(QuantizationInfo(*outputScales.data, 0));

    arm_compute::Tensor& src = info->src;
    arm_compute::Tensor& weights = info->weights;
    arm_compute::Tensor& dst = info->dst;

    // Initialize tensors
    src.allocator()->init(src_info);
    weights.allocator()->init(weights_info);
    dst.allocator()->init(dst_info);

    info->conv.configure(
        &src, &weights, nullptr, &dst,
        PadStrideInfo{params.strides[1], params.strides[0], params.padding_l[1],
                      params.padding_r[1], params.padding_l[0],
                      params.padding_r[0], DimensionRoundingType::FLOOR},
        WeightsInfo{}, Size2D{params.dilates[1], params.dilates[0]});

    return info;
  };

  std::shared_ptr<AclConvInfo> info;
  if (isWeightPrePackingEnabled() && params.weight_is_const) {
    std::string unique_name = "tao_ral.cpu.acl_qconv_s8s8s8";
    auto state = ctx->getOrCreateResource<ACLQConvState>(
        unique_name, []() { return new ACLQConvState; });
    {
      ConvParamsKey key;
      key.src_dims.reserve(NDims);
      key.weight_dims.reserve(NDims);
      key.dst_dims.reserve(NDims);
      key.metadatas.reserve(metadata.sizes[0] + padding.sizes[0]);
      for (int i = 0; i < NDims; ++i) {
        key.src_dims.push_back(input.sizes[i]);
        key.weight_dims.push_back(kernel.sizes[i]);
        key.dst_dims.push_back(output.sizes[i]);
      }
      for (int i = 0; i < metadata.sizes[0]; ++i) {
        key.metadatas.push_back(metadata.data[i]);
      }
      for (int i = 0; i < padding.sizes[0]; ++i) {
        key.metadatas.push_back(padding.data[i]);
      }
      key.weight_ptr = kernel.data;
      key.tid = std::this_thread::get_id();
      std::lock_guard<std::mutex> l(state->mu);
      auto& cache = state->cache;
      auto it = cache.find(key);
      if (it == cache.end()) {
        it = cache.insert(std::make_pair(key, AclConvCreator())).first;
      }
      info = it->second;
    }
  } else {
    info = AclConvCreator();
  }

  arm_compute::Tensor& src = info->src;
  arm_compute::Tensor& weights = info->weights;
  arm_compute::Tensor& dst = info->dst;

  // DataLayout data_layout = DataLayout::NHWC;
  // TensorShape src_shape, weights_shape, biases_shape, dst_shape;
  // src_shape = TensorShape(Ci, W, H, N);
  // weights_shape = TensorShape(Ci, Kw, Kh, Co);
  // dst_shape = TensorShape(Co, W, H, N);

  // DataType data_type = DataType::QASYMM8_SIGNED;
  // TensorInfo src_info = TensorInfo(src_shape, 1, data_type, data_layout);
  // TensorInfo weights_info =
  //     TensorInfo(weights_shape, 1, DataType::QSYMM8_PER_CHANNEL,
  //     data_layout);
  // TensorInfo dst_info = TensorInfo(dst_shape, 1, data_type, data_layout);

  // const QuantizationInfo src_qinfo = QuantizationInfo();
  // src_info.set_quantization_info(QuantizationInfo(*inputScales.data, 0));
  // std::vector<float> scales(filterScales.data,
  //                           filterScales.data + filterScales.sizes[0]);
  // weights_info.set_quantization_info(QuantizationInfo(std::move(scales)));
  // dst_info.set_quantization_info(QuantizationInfo(*outputScales.data, 0));

  // arm_compute::Tensor src, weights, dst;
  // // Initialize tensors
  // src.allocator()->init(src_info);
  // weights.allocator()->init(weights_info);
  // dst.allocator()->init(dst_info);

  src.allocator()->import_memory(input.data);
  weights.allocator()->import_memory(kernel.data);
  dst.allocator()->import_memory(output.data);

  // NEConvolutionLayer conv{};
  // conv.configure(
  //     &src, &weights, nullptr, &dst,
  //     PadStrideInfo{params.strides[1], params.strides[0],
  //     params.padding_l[1],
  //                   params.padding_r[1], params.padding_l[0],
  //                   params.padding_r[0], DimensionRoundingType::FLOOR},
  //     WeightsInfo{}, Size2D{params.dilates[1], params.dilates[0]});

  info->conv.run();

  timer.Stop();
  if (isProfilingEnabled()) {
    const auto& src_dims = params.src.get_dims();
    const auto& kernel_dims = params.weight.get_dims();
    const auto& dst_dims = params.dst.get_dims();

    int64_t bytes =
        static_cast<int64_t>(params.src.get_nelems()) * sizeof(int8_t) +
        static_cast<int64_t>(params.weight.get_nelems()) * sizeof(int8_t) +
        static_cast<int64_t>(params.dst.get_nelems()) * sizeof(int8_t);

    // N * OC * OH * OW * KH * KW * KIC
    int64_t gflops = 2 * static_cast<int64_t>(params.dst.get_nelems()) *
                     static_cast<int64_t>(params.weight.get_nelems()) /
                     kernel_dims[0];

    std::ostringstream sout;
    sout << "ral_qconv_s8_s8_s8:\n";
    sout << "  input logical NCHW shape:\n\t";
    for (const auto& d : src_dims) {
      sout << d << " ";
    }
    sout << "\n  kernel logical OIHW shape:\n\t";
    for (const auto& d : kernel_dims) {
      sout << d << " ";
    }
    sout << "\n  output logical NCHW shape:\n\t";
    for (const auto& d : dst_dims) {
      sout << d << " ";
    }
    sout << "\n  strides:\n\t";
    for (size_t i = 0; i < params.strides.size(); ++i) {
      sout << params.strides[i] << " ";
    }
    sout << "\n  dilates:\n\t";
    for (size_t i = 0; i < params.dilates.size(); ++i) {
      sout << params.dilates[i] << " ";
    }
    sout << "\n  paddings_l:\n\t";
    for (size_t i = 0; i < params.padding_l.size(); ++i) {
      sout << params.padding_l[i] << " ";
    }
    sout << "\n  paddings_r:\n\t";
    for (size_t i = 0; i < params.padding_r.size(); ++i) {
      sout << params.padding_r[i] << " ";
    }
    TAO_VLOG(0) << sout.str() << "\n roofline:\n"
                << "\tMath Ops = " << gflops << "\n"
                << "\tBytes = " << bytes << "\n"
                << "\tBandwidth = "
                << double(bytes) / double(timer.GetNanoSeconds()) << " GB\n"
                << "\tGFLOPS = "
                << double(gflops) / double(timer.GetNanoSeconds()) << "\n";
  }
}

}  // namespace

TAO_RAL_API("ral_qconv_s8_s8_s8", "cpu", ral_qconv_s8_s8_s8<4>);

}  // namespace ral
}  // namespace tao
#endif
