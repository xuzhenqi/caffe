// --------------------------------------------------------
// Fast R-CNN
// Copyright (c) Microsoft. All rights reserved.
// Written by Ross Girshick, 2015.
// Licensed under the BSD 2-clause "Simplified" license.
// See LICENSE in the Fast R-CNN project root for license
// information.
// --------------------------------------------------------

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
  SmoothL1LossParameter smooth_l1_loss_param = this->layer_param_.smooth_l1_loss_param();
  
  CHECK_GT(smooth_l1_loss_param.threshold(), 0)
  << "\n Threshold must be greater than 0.";
  CHECK_GE(smooth_l1_loss_param.margin(), 0)
  << "\n Margin must be not less than 0.";

  th=smooth_l1_loss_param.threshold();
  margin=smooth_l1_loss_param.margin();
  has_weights_ = (bottom.size() == 3);

  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe
