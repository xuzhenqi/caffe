#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductRNNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(3);
    } else {
      this->blobs_.resize(2);
    }
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    weight_shape[1] = N_;
    this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[1].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[2].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  Reshape(bottom, top);
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    // NOLINT_NEXT_LINE(caffe/alt_fn)
    caffe_gpu_set(previous_.count(), Dtype(0), previous_.mutable_gpu_data());
#else
    NO_GPU;
#endif
  } else {
    caffe_set(previous_.count(), Dtype(0), previous_.mutable_cpu_data());
  }
  neuron_layer_->LayerSetUp(top, top);
}

template <typename Dtype>
void InnerProductRNNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  neuron_layer_->Reshape(top, top);
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  previous_.ReshapeLike(*(top[0]));
  previous_out_.ReshapeLike(*(top[0]));
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }

  CHECK_EQ(bottom[1]->count(), bottom[0]->num());
}

template <typename Dtype>
void InnerProductRNNLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[2]->cpu_data(), (Dtype)1., top_data);
  }
  bottom_data = previous_.cpu_data();
  top_data = previous_out_.mutable_cpu_data();
  weight = this->blobs_[1]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  caffe_add(previous_.count(), top[0]->cpu_data(), previous_out_.cpu_data(),
            top[0]->mutable_cpu_data());
  neuron_layer_->Forward(top, top);
  caffe_copy(previous_.count(), previous_.cpu_data(), 
             previous_out_.mutable_cpu_data());
  caffe_copy(previous_.count(), top[0]->cpu_data(), 
             previous_.mutable_cpu_data());
  const Dtype* end_mark = bottom[1]->cpu_data();
  int dim = previous_.count() / previous_.num();
  for (int i = 0; i < bottom[1]->count(); ++i) {
    if (end_mark[i] > 0.5) {
      caffe_set(dim, Dtype(0), 
                previous_.mutable_cpu_data() + previous_.offset(i));
    }
  }
}

template <typename Dtype>
void InnerProductRNNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  vector<bool> temp(top.size(), true);
  neuron_layer_->Backward(top, temp, top);
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }
  if (this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = previous_out_.cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, N_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[1]->mutable_cpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[2]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[2]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductRNNLayer);
#endif

INSTANTIATE_CLASS(InnerProductRNNLayer);
REGISTER_LAYER_CLASS(InnerProductRNN);

}  // namespace caffe
