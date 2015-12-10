#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void caffe_gpu_max(const int N, const Dtype alpha, const Dtype *x,
                              Dtype *y) {
  CUDA_KERNEL_LOOP(index, N) {
    if (x[index] < alpha)
      y[index] = alpha;
  }
}



template <typename Dtype>
void SoftmaxEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob < Dtype> *> & bottom,
    const vector<Blob < Dtype> *> & top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  Dtype* prob_data = prob_.mutable_gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  Dtype *temp = prob_.mutable_gpu_diff();
  const int count = prob_.count();
  
  caffe_gpu_max<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, Dtype(FLT_MIN), prob_data, prob_data);
  caffe_gpu_log(count, prob_data, temp);
  caffe_gpu_dot(count, temp, label, top[0]->mutable_cpu_data());
  top[0]->mutable_cpu_data()[0] /= -outer_num_;
}

template <typename Dtype>
void SoftmaxEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int count = prob_.count();
    const Dtype *label = bottom[1]->gpu_data();
    const Dtype *prob_data = prob_.gpu_data();
    Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_sub(count, prob_data, label, bottom_diff);
    caffe_gpu_scal(count, top[0]->cpu_diff()[0] / outer_num_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxEntropyLossLayer);

} // namespace caffe
