#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2NLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  eps_ = this->layer_param_.l2n_param().eps();
}

template <typename Dtype>
void L2NLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  if (top[0] == bottom[0]) {
    LOG(FATAL) << "L2NLayer doesn't support in-place computation.";
  }
  top[0]->ReshapeLike(*bottom[0]);
  temp_.ReshapeLike(*bottom[0]);
  num_ = bottom[0]->num();
  dimension_ = bottom[0]->count() / num_;
  norm_.Reshape(num_, 1, 1, 1);
  multiplier_.Reshape(dimension_, 1, 1, 1);
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    caffe_gpu_set(dimension_, Dtype(1), multiplier_.mutable_gpu_data());
#else
    NO_GPU;
#endif
  } else {
    caffe_set(dimension_, Dtype(1), multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void L2NLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  int size = num_ * dimension_;
  caffe_sqr(size, bottom[0]->cpu_data(), temp_.mutable_cpu_data()); // x^2 
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_, dimension_, Dtype(1), 
          temp_.cpu_data(), multiplier_.cpu_data(), Dtype(0.), 
          norm_.mutable_cpu_data()); 
  // sum(x^2)
  caffe_sqrt(num_, norm_.cpu_data(), norm_.mutable_cpu_data()); // norm
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, dimension_, 1, 
           Dtype(1), norm_.cpu_data(), multiplier_.cpu_data(), Dtype(0), 
           temp_.mutable_cpu_data());
  caffe_add_scalar(size, eps_, temp_.mutable_cpu_data());
  caffe_div(size, bottom[0]->cpu_data(), temp_.cpu_data(), 
            top[0]->mutable_cpu_data()); // x./(norm + eps)
}

template <typename Dtype>
void L2NLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]){
    int size = num_ * dimension_;
    caffe_mul(size, bottom[0]->cpu_data(), top[0]->cpu_diff(), 
          temp_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num_, dimension_, Dtype(1.), 
          temp_.cpu_data(), multiplier_.cpu_data(), Dtype(0.),
          norm_.mutable_cpu_diff());
    caffe_cub(num_, norm_.cpu_data(), temp_.mutable_cpu_data());
    caffe_add_scalar(num_, eps_, temp_.mutable_cpu_data());
    caffe_div(num_, norm_.cpu_diff(), temp_.cpu_data(), 
              norm_.mutable_cpu_diff());
    caffe_set(num_, Dtype(1.), temp_.mutable_cpu_data());
    caffe_add_scalar(num_, eps_, norm_.mutable_cpu_data());
    caffe_div(num_, temp_.cpu_data(), norm_.cpu_data(), 
              norm_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, dimension_, 1, Dtype(1.),
            norm_.cpu_data(), multiplier_.cpu_data(), Dtype(0.),
            temp_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, dimension_, 1, 
            Dtype(1.), norm_.cpu_diff(), multiplier_.cpu_data(), Dtype(0.),
            temp_.mutable_cpu_diff());
    caffe_mul(size, temp_.cpu_data(), top[0]->cpu_diff(),
              temp_.mutable_cpu_data());
    caffe_mul(size, temp_.cpu_diff(), bottom[0]->cpu_data(),
              temp_.mutable_cpu_diff());
    caffe_sub(size, temp_.cpu_data(), temp_.cpu_diff(),
              bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
  STUB_GPU(L2NLayer);
#endif

  INSTANTIATE_CLASS(L2NLayer);
  REGISTER_LAYER_CLASS(L2N);

}  // namespace caffe
