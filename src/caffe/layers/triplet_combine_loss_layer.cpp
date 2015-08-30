#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
void TripletCombineLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  margin_ = this->layer_param_.triplet_combine_loss_param().margin();
  class_ = this->layer_param_.triplet_combine_loss_param().classes();
  num_ = this->layer_param_.triplet_combine_loss_param().num();
}

template <typename Dtype>
void TripletCombineLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), num_ * class_);
  dim_ = bottom[0]->count() / num_ / class_;
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  diff_.reset(new SyncedMemory(num_ * class_ * num_ * class_ * 
                               dim_ * sizeof(Dtype)));
  dist_.reset(new SyncedMemory(num_ * class_ * num_ * class_ *
                               sizeof(Dtype)));
  loss_.reset(new SyncedMemory(num_ * class_ * num_ * class_ * num_ 
                               * class_ * sizeof(Dtype)));
}

template <typename Dtype>
void TripletCombineLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  // Compute diff_
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  Dtype *diff_data = (Dtype*)diff_->mutable_cpu_data();
  for (int i = 0; i < num_ * class_; ++i){
    for (int j = 0; j < num_ * class_; ++j) {
      if (i == j)
        continue;
      caffe_sub(dim_, bottom_data + dim_ * i, bottom_data + dim_ * j,
                diff_data + (i * num_ + j) * dim_);
    }
  }
  // Compute dist_
  Dtype *dist_data = (Dtype*)dist_->mutable_cpu_data();
  for (int i = 0; i < num_ * class_ - 1; ++i) {
    for (int j = i; j < num_ * class_; ++j) {
      dist_data[i * num_ + j] = 0;
      for (int k = 0; k < dim_; ++k) {
        dist_data[i * num_ + j] += diff_data[(i * num_ + j) * dim_ + k] *
            diff_data[(i * num_ + j) * dim_ + k];
      }
      dist_data[i * num_ + j] = sqrt(dist_data[i * num_ + j]);
      dist_data[j * num_ + i] = dist_data[i * num_ + j];
    }
  }
  // Compute loss_ and loss
  Dtype *loss_data = (Dtype *)loss_->cpu_data();
  diff_data = (Dtype *)diff_->cpu_data();
  int count = 0;
  int loss = 0;
  int size = num_ * class_;
  int temp_loss;
  for (int i = 0; i < class_; ++i) {
    for (int j = i * num_; j < (i + 1) * num_ - 1; ++j) {
      for (int k = j; k < (i + 1) * num_; ++k) {
        for (int l = 0; l < class_ * num_; ++l) {
          if (l == i * num_) {
            l += num_ - 1;
            continue;
          }
          count += 2;
          temp_loss = dist_data[j * size + k] + margin_ - 
              dist_data[j * size + l];
          if (temp_loss > 0)
            loss += temp_loss;
          loss_data[(j * size + k) * size + l] = temp_loss;
          temp_loss = dist_data[j * size + k] + margin_ - 
              dist_data[k * size + l];
          if (temp_loss > 0)
            loss += temp_loss;
          loss_data[(k * size + j) * size + l] = temp_loss;
        }
      }
    }
  }
  CHECK_EQ(count, size * (num_ - 1) * (class_ - 1) * num_);
  top_data[0] = loss / count;
}

template <typename Dtype>
void TripletCombineLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    
  }
}

#ifdef CPU_ONLY
STUB_GPU(TripletCombineLossLayer);
#endif

INSTANTIATE_CLASS(TripletCombineLossLayer);
REGISTER_LAYER_CLASS(TripletCombineLoss);

} // namespace caffe
