#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  margin_ = this->layer_param_.triplet_loss_param().margin();
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  CHECK_EQ(num_, bottom[1]->num());
  CHECK_EQ(num_, bottom[2]->num());
  dim_ = bottom[0]->count();
  CHECK_EQ(dim_, bottom[1]->count());
  CHECK_EQ(dim_, bottom[2]->count());
  dim_ /= num_;
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  p_sub_a_.ReshapeLike(*bottom[0]);
  a_sub_n_.ReshapeLike(*bottom[0]);
  temp_.ReshapeLike(*bottom[0]);
  if (Caffe::mode() == Caffe::GPU){
    caffe_gpu_set(num_*dim_, Dtype(1.), temp_.mutable_gpu_diff());
  } else {
    caffe_set(num_*dim_, Dtype(1.), temp_.mutable_cpu_diff());
  }
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  // bottom[0] : anchor
  // bottom[1] : positive
  // bottom[2] : negative
  // Outcome: 
  // p_sub_a_.data : p - a
  // a_sub_n_.data : a - n
  // temp_.data[1..num_] : distance of sample 1..num_.
  // p_sub_a_.diff and a_sub_n_.diff are used as temperol buffers.
  int size = num_ * dim_;
  caffe_sub(size, bottom[1]->cpu_data(), bottom[0]->cpu_data(), 
            p_sub_a_.mutable_cpu_data());
  caffe_sub(size, bottom[0]->cpu_data(), bottom[2]->cpu_data(),
            a_sub_n_.mutable_cpu_data());
  caffe_sqr(size, p_sub_a_.cpu_data(), p_sub_a_.mutable_cpu_diff());
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_, dim_, Dtype(1.), p_sub_a_.cpu_diff(),
            temp_.cpu_diff(), Dtype(0.), temp_.mutable_cpu_data());
  caffe_sqr(size, a_sub_n_.cpu_data(), a_sub_n_.mutable_cpu_diff());
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_, dim_, Dtype(1.), a_sub_n_.cpu_diff(),
            temp_.cpu_diff(), Dtype(0.), p_sub_a_.mutable_cpu_diff());
  caffe_sub(num_, temp_.cpu_data(), p_sub_a_.cpu_diff(), 
            temp_.mutable_cpu_data());
  caffe_add_scalar(num_, margin_, temp_.mutable_cpu_data());
  caffe_cpu_max(num_, temp_.cpu_data(), Dtype(0.), p_sub_a_.mutable_cpu_diff());
  caffe_cpu_gemv<Dtype>(CblasNoTrans, 1, num_, Dtype(0.5/num_), 
            p_sub_a_.cpu_diff(), temp_.cpu_diff(), Dtype(0.), 
            top[0]->mutable_cpu_data());
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int size = num_ * dim_;
  const Dtype alpha = top[0]->cpu_diff()[0] / num_;
  caffe_cpu_large(num_, temp_.cpu_data(), Dtype(0.), temp_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, dim_, 1, alpha, 
            temp_.cpu_data(), temp_.cpu_diff(), Dtype(0.), 
            p_sub_a_.mutable_cpu_diff());
  if (propagate_down[0]) {
    caffe_sub(size, bottom[2]->cpu_data(), bottom[1]->cpu_data(), 
              a_sub_n_.mutable_cpu_diff());
    caffe_mul(size, p_sub_a_.cpu_diff(), a_sub_n_.cpu_diff(), 
              bottom[0]->mutable_cpu_diff());
  }
  if (propagate_down[1]) {
    caffe_mul(size, p_sub_a_.cpu_diff(), p_sub_a_.cpu_data(),
              bottom[1]->mutable_cpu_diff());
  }
  if (propagate_down[2]) {
    caffe_mul(size, p_sub_a_.cpu_diff(), a_sub_n_.cpu_data(),
              bottom[2]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

} // namespace caffe
