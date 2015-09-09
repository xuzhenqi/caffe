#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductRNNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[2]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[2]->gpu_data(), (Dtype)1., top_data);
  }
  bottom_data = previous_.gpu_data();
  top_data = previous_out_.mutable_gpu_data();
  weight = this->blobs_[1]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, N_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
  }
  caffe_gpu_add(previous_.count(), top[0]->gpu_data(), 
                previous_out_.gpu_data(),
                top[0]->mutable_gpu_data());
  neuron_layer_->Forward(top, top);
  caffe_copy(previous_.count(), previous_.gpu_data(), 
             previous_out_.mutable_gpu_data());
  caffe_copy(previous_.count(), top[0]->gpu_data(), 
             previous_.mutable_gpu_data());
  const Dtype* end_mark = bottom[1]->cpu_data();
  int dim = previous_.count() / previous_.num();
  for (int i = 0; i < bottom[1]->count(); ++i) {
    if (end_mark[i] > 0.5) {
      caffe_gpu_set(dim, Dtype(0), 
                previous_.mutable_gpu_data() + previous_.offset(i));
    }
  }

}

template <typename Dtype>
void InnerProductRNNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  vector<bool> temp(top.size(), true);
  neuron_layer_->Backward(top, temp, top);
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
  }
  if (this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = previous_out_.gpu_data();
    // Gradient with respect to weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, N_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[1]->mutable_gpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[2]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[2]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductRNNLayer);

}  // namespace caffe
