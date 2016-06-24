// --------------------------------------------------------
// Fast R-CNN
// Copyright (c) Microsoft. All rights reserved.
// Written by Ross Girshick, 2015.
// Licensed under the BSD 2-clause "Simplified" license.
// See LICENSE in the Fast R-CNN project root for license
// information.
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SmoothL1ForwardGPU(const int n, const Dtype th, const Dtype margin, const Dtype* in, Dtype* out) {
    // f(x) = 0                              if |x| < margin
    //        0.5 * (|x| - margin)^2 / th    if margin <= |x| < th + margin
    //        |x| - margin - 0.5 * th        otherwise
    CUDA_KERNEL_LOOP(index, n) {
        Dtype val = in[index];
        Dtype abs_val = abs(val);
        if (abs_val < margin) {
            out[index] = 0;
        }
        else if(abs_val < margin + th) {
            out[index] = 0.5 * (abs_val-margin) * (abs_val-margin) / th;
        }
        else {
            out[index] = abs_val - margin - 0.5 * th;
        }
    }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();
    caffe_gpu_sub(
        count,
        bottom[0]->gpu_data(),
        bottom[1]->gpu_data(),
        diff_.mutable_gpu_data());    // d := b0 - b1
    if (has_weights_) {
        caffe_gpu_mul(
            count,
            bottom[2]->gpu_data(),
            diff_.gpu_data(),
            diff_.mutable_gpu_data());  // d := w * (b0 - b1)
    }
    SmoothL1ForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
        count, th, margin, diff_.gpu_data(), errors_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

    Dtype loss;
    caffe_gpu_asum(count, errors_.gpu_data(), &loss);
    top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num(); // This is the original implementation in *Fast* R-CNN
}

template <typename Dtype>
__global__ void SmoothL1BackwardGPU(const int n, const Dtype th, const Dtype margin, const Dtype* in, Dtype* out) {
    // f'(x) = 0                               if |x| < margin
    //       = sign(x) * (|x| - margin) / th   if margin <= |x| < th + margin
    //       = sign(x)                         otherwise
    CUDA_KERNEL_LOOP(index, n) {
        Dtype val = in[index];
        Dtype sign = val>Dtype(0)?Dtype(1):Dtype(-1);
        Dtype abs_val = val*sign;
        if (abs_val < margin) {
            out[index] = 0;
        }
        else if(abs_val < margin + th) {
            out[index] = sign * (abs_val-margin) / th;
        }
        else {
            out[index] = sign;
        }
    }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    int count = diff_.count();
    SmoothL1BackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
        count, th, margin, diff_.gpu_data(), diff_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    for (int i = 0; i < 2; ++i) {
        if (propagate_down[i]) {
            const Dtype sign = (i == 0) ? 1 : -1;
            int spatial_dim = diff_.height() * diff_.width();
            const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
            caffe_gpu_axpby(
                bottom[i]->count(),              // count
                alpha,                           // alpha
                diff_.gpu_data(),                // x
                Dtype(0),                        // beta
                bottom[i]->mutable_gpu_diff());  // y
        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(SmoothL1LossLayer);

}  // namespace caffe
