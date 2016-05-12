#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KLDivergenceLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype>*>&top) {
  const int count = bottom[0]->count();
  const Dtype* pred = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  Dtype* buffer_pred = buffer_.mutable_gpu_data();
  Dtype* buffer_label = buffer_.mutable_gpu_diff();
  caffe_copy(count, pred, buffer_pred);
  caffe_copy(count, label, buffer_label);
  caffe_gpu_add_scalar(count, margin_, buffer_pred);
  caffe_gpu_add_scalar(count, margin_, buffer_label);
  caffe_gpu_div(count, buffer_label, buffer_pred, buffer_label);
  caffe_gpu_log(count, buffer_label, buffer_label);
  caffe_gpu_dot(count, label, buffer_label, top[0]->mutable_cpu_data());
  top[0]->mutable_cpu_data()[0] /= outer_num_;
}

template <typename Dtype>
void KLDivergenceLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>&top, const vector<bool>&propagate_down,
    const vector<Blob<Dtype>*>&bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const Dtype* label = bottom[1]->gpu_data();
    const Dtype* pred_buffer = buffer_.gpu_data();
    Dtype* pred_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_div(count, label, pred_buffer, pred_diff);
    caffe_gpu_scal(count, -top[0]->cpu_diff()[0] / outer_num_, pred_diff);
  } 
}

INSTANTIATE_LAYER_GPU_FUNCS(KLDivergenceLossLayer);

}  // namespace caffe
