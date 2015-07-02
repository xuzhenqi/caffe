#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiConstructionLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  const int size = bottom.size();
  CHECK_EQ(bottom[0]->count(), size - 2);
  for(int i=2; i<size; ++i){
    CHECK_EQ(bottom[1]->count(), bottom[i]->count());
  }
  temp_.Reshape(bottom[1]->shape());
  gap_.Reshape(bottom[1]->shape());
  inner_.Reshape(1, 1, (size-2)*(size-3)/2, bottom[1]->num());
  ones_.Reshape(bottom[1]->shape());
  caffe_set(bottom[1]->count(), Dtype(1), ones_.mutable_cpu_data());
  alpha_ = this->layer_param_.multi_construction_loss_param().alpha();

}

template <typename Dtype>
void MultiConstructionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[1]->count();
  const Dtype *label = bottom[0]->cpu_data();
  caffe_set(count, Dtype(0), gap_.mutable_cpu_data());
  for (int i=2; i<bottom.size(); ++i){
    caffe_cpu_axpby(count, label[i-2], bottom[i]->cpu_data(), Dtype(1), 
                    gap_.mutable_cpu_data());
  }
  caffe_sub(count, gap_.cpu_data(), bottom[1]->cpu_data(), 
            gap_.mutable_cpu_data());
  int num = temp_.num();
  int channel = temp_.channels();
  int height = temp_.height();
  int width = temp_.width();
  int dim = channel * height * width;
  int j = 0;
  for(int i=2; i<bottom.size(); ++i){
    for(int k=i+1; k<bottom.size(); ++k){
      caffe_mul(count, bottom[i]->cpu_data(), bottom[k]->cpu_data(),
                temp_.mutable_cpu_data());
      caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, Dtype(1), temp_.cpu_data(),
            ones_.cpu_data(), Dtype(0), inner_.mutable_cpu_data() + (j++) * num);
    }
  }
  top[0]->mutable_cpu_data()[0] = 
      (gap_.sumsq_data() + inner_.sumsq_data() * alpha_) / num;
}

template <typename Dtype>
void MultiConstructionLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down, 
    const vector<Blob<Dtype>*>& bottom){
  int count = bottom[1]->count();
  Dtype *label = bottom[0]->mutable_cpu_data();
  
  int num = temp_.num();
  int channel = temp_.channels();
  int height = temp_.height();
  int width = temp_.width();
  int dim = channel * height * width;

  caffe_set(count, Dtype(0), temp_.mutable_cpu_data());
  for(int i=2; i<bottom.size(); ++i){
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
        inner_.cpu_data() + (i++) * num, ones_.cpu_data(), 0, 
        temp_.mutable_cpu_diff());
    caffe_mul(count, bottom[i]->cpu_data(), temp_.cpu_diff(), 
              gap_.mutable_cpu_diff());
    caffe_add(count, gap_.cpu_diff(), temp_.cpu_data(), 
              temp_.mutable_cpu_data());
  }

  for(int i=2; i<bottom.size(); ++i){
    caffe_copy(count, gap_.cpu_data(), bottom[i]->mutable_cpu_diff());
    caffe_scal(count, label[i-2] / num, bottom[i]->mutable_cpu_diff());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
        inner_.cpu_data() + (i++) * num, ones_.cpu_data(), 0, 
        temp_.mutable_cpu_diff());
    caffe_mul(count, bottom[i]->cpu_data(), temp_.cpu_diff(), 
              gap_.mutable_cpu_diff());
    caffe_sub(count, temp_.cpu_data(), gap_.cpu_diff(), 
              temp_.mutable_cpu_data());
    caffe_scal(count, alpha_ / num, temp_.mutable_cpu_data());
    caffe_add(count, bottom[i]->cpu_diff(), temp_.cpu_data(),
              bottom[i]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiConstructionLossLayer);
#endif

INSTANTIATE_CLASS(MultiConstructionLossLayer);
REGISTER_LAYER_CLASS(MultiConstructionLoss);

} // namespace caffe
