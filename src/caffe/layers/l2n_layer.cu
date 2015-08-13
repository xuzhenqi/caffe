#include <cublas_v2.h>

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void L2NLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  int size = num_ * dimension_;
  caffe_gpu_sqr(size, bottom[0]->gpu_data(), temp_.mutable_gpu_data()); // x^2
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_, dimension_, Dtype(1.),
          temp_.gpu_data(), multiplier_.gpu_data(), Dtype(0.),
          norm_.mutable_gpu_data());
  // sum(x^2)
  caffe_gpu_sqrt(num_, norm_.gpu_data(), norm_.mutable_gpu_data()); // norm
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, dimension_, 1, 
          Dtype(1.), norm_.gpu_data(), multiplier_.gpu_data(), Dtype(0.),
          temp_.mutable_gpu_data());
  caffe_gpu_add_scalar(size, eps_, temp_.mutable_gpu_data());
  caffe_gpu_div(size, bottom[0]->gpu_data(), temp_.gpu_data(),
                top[0]->mutable_gpu_data()); // x./(norm + eps)
}

template <typename Dtype>
void L2NLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]){
    int size = num_ * dimension_;
    caffe_gpu_mul(size, bottom[0]->gpu_data(), top[0]->gpu_diff(), 
          temp_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_, dimension_, Dtype(1.), 
          temp_.gpu_data(), multiplier_.gpu_data(), Dtype(0.),
          norm_.mutable_gpu_diff());
    caffe_gpu_cub(num_, norm_.gpu_data(), temp_.mutable_gpu_data());
    caffe_gpu_add_scalar(num_, eps_, temp_.mutable_gpu_data());
    caffe_gpu_div(num_, norm_.gpu_diff(), temp_.gpu_data(), 
              norm_.mutable_gpu_diff());
    caffe_gpu_set(num_, Dtype(1.), temp_.mutable_gpu_data());
    caffe_gpu_add_scalar(num_, eps_, norm_.mutable_gpu_data());
    caffe_gpu_div(num_, temp_.gpu_data(), norm_.gpu_data(), 
              norm_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, dimension_, 1, 
            Dtype(1.), norm_.gpu_data(), multiplier_.gpu_data(), Dtype(0.),
            temp_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, dimension_, 1, 
            Dtype(1.), norm_.gpu_diff(), multiplier_.gpu_data(), Dtype(0.),
            temp_.mutable_gpu_diff());
    caffe_gpu_mul(size, temp_.gpu_data(), top[0]->gpu_diff(),
              temp_.mutable_gpu_data());
    caffe_gpu_mul(size, temp_.gpu_diff(), bottom[0]->gpu_data(),
              temp_.mutable_gpu_diff());
    caffe_gpu_sub(size, temp_.gpu_data(), temp_.gpu_diff(),
              bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(L2NLayer);

}  // namespace caffe
