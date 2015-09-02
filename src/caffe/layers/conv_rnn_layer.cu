#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int n = 0; n < this->num_; ++n) {
    this->forward_gpu_gemm(bottom_data + bottom[0]->offset(n), weight,
        top_data + top[0]->offset(n));
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[2]->gpu_data();
      this->forward_gpu_bias(top_data + top[0]->offset(n), bias);
    }
  }
  weight = this->blobs_[1]->gpu_data();
  top_data = previous_out_.mutable_gpu_data();
  bottom_data = previous_.gpu_data();
  for (int n = 0; n < this->num_; ++n) {
    this->forward_rnn_gpu_gemm(bottom_data + previous_.offset(n), weight,
        top_data + previous_out_.offset(n));
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[3]->gpu_data();
      this->forward_gpu_bias(top_data + previous_out_.offset(n), bias);
    }
  }
  caffe_gpu_add(previous_.count(), previous_out_.gpu_data(), top[0]->gpu_data(),
            top[0]->mutable_gpu_data());
  caffe_copy(previous_.count(), previous_.gpu_data(), 
             previous_out_.mutable_gpu_data());
  caffe_copy(previous_.count(), top[0]->gpu_data(), 
             previous_.mutable_gpu_data());
}

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[2]) {
    Dtype* bias_diff = this->blobs_[2]->mutable_gpu_diff();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_gpu_bias(bias_diff, top_diff + top[0]->offset(n));
    }
  }
  if (this->param_propagate_down_[0] || propagate_down[0]) {
    for (int n = 0; n < this->num_; ++n) {
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0]) {
        this->weight_gpu_gemm(bottom_data + bottom[0]->offset(n),
            top_diff + top[0]->offset(n), weight_diff);
      }
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[0]) {
        this->backward_gpu_gemm(top_diff + top[0]->offset(n), weight,
            bottom_diff + bottom[0]->offset(n));
      }
    }
  }
  
  weight = this->blobs_[1]->gpu_data();
  weight_diff = this->blobs_[1]->mutable_gpu_diff();
  top_diff = top[0]->gpu_diff();
  bottom_data = previous_out_.gpu_data();
  // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[3]) {
    Dtype* bias_diff = this->blobs_[3]->mutable_gpu_diff();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_gpu_bias(bias_diff, top_diff + top[0]->offset(n));
    }
  }
  if (this->param_propagate_down_[1] || propagate_down[0]) {
    for (int n = 0; n < this->num_; ++n) {
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[1]) {
        this->weight_rnn_gpu_gemm(bottom_data + previous_out_.offset(n),
            top_diff + top[0]->offset(n), weight_diff);
      }
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionRNNLayer);

}  // namespace caffe
