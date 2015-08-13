#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // bottom[0] : anchor
  // bottom[1] : positive
  // bottom[2] : negative
  // Outcome: 
  // p_sub_a_.data : p - a
  // a_sub_n_.data : a - n
  // temp_.data[1..num_] : distance of sample 1..num_.
  // p_sub_a_.diff and a_sub_n_.diff are used as temperol buffers.
  int size = num_ * dim_;
  caffe_gpu_sub(size, bottom[1]->gpu_data(), bottom[0]->gpu_data(), 
            p_sub_a_.mutable_gpu_data());
  caffe_gpu_sub(size, bottom[0]->gpu_data(), bottom[2]->gpu_data(),
            a_sub_n_.mutable_gpu_data());
  caffe_gpu_sqr(size, p_sub_a_.gpu_data(), p_sub_a_.mutable_gpu_diff());
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_, dim_, Dtype(1.), p_sub_a_.gpu_diff(),
            temp_.gpu_diff(), Dtype(0.), temp_.mutable_gpu_data());
  caffe_gpu_sqr(size, a_sub_n_.gpu_data(), a_sub_n_.mutable_gpu_diff());
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_, dim_, Dtype(1.), a_sub_n_.gpu_diff(),
            temp_.gpu_diff(), Dtype(0.), p_sub_a_.mutable_gpu_diff());
  caffe_gpu_sub(num_, temp_.gpu_data(), p_sub_a_.gpu_diff(), 
            temp_.mutable_gpu_data());
  caffe_gpu_add_scalar(num_, margin_, temp_.mutable_gpu_data());
  caffe_gpu_max(num_, temp_.gpu_data(), Dtype(0.), p_sub_a_.mutable_gpu_diff());
  caffe_gpu_gemv<Dtype>(CblasNoTrans, 1, num_, Dtype(0.5/num_), 
            p_sub_a_.gpu_diff(), temp_.gpu_diff(), Dtype(0.), 
            top[0]->mutable_gpu_data());
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int size = num_ * dim_;
  const Dtype alpha = top[0]->cpu_diff()[0] / num_;
  caffe_gpu_large(num_, temp_.gpu_data(), Dtype(0.), temp_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, dim_, 1, alpha, 
            temp_.gpu_data(), temp_.gpu_diff(), Dtype(0.), 
            p_sub_a_.mutable_gpu_diff());
  if (propagate_down[0]) {
    caffe_gpu_sub(size, bottom[2]->gpu_data(), bottom[1]->gpu_data(), 
              a_sub_n_.mutable_gpu_diff());
    caffe_gpu_mul(size, p_sub_a_.gpu_diff(), a_sub_n_.gpu_diff(), 
              bottom[0]->mutable_gpu_diff());
  }
  if (propagate_down[1]) {
    caffe_gpu_mul(size, p_sub_a_.gpu_diff(), p_sub_a_.gpu_data(),
              bottom[1]->mutable_gpu_diff());
  }
  if (propagate_down[2]) {
    caffe_gpu_mul(size, p_sub_a_.gpu_diff(), a_sub_n_.gpu_data(),
              bottom[2]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);

}  // namespace caffe
