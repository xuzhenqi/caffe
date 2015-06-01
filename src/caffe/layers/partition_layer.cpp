#include<vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PartitionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  const int size = this->layer_param_.partition_param_size();
  CHECK(size > 0) << "Please give at least one partion_param.";
  left_.clear();
  right_.clear();
  up_.clear();
  down_.clear();
  for(int i=0; i<size; ++i){
    const PartitionParameter &pp = this->layer_param_.partition_param(i);
    left_.push_back(pp.left());
    right_.push_back(pp.right());
    up_.push_back(pp.up());
    down_.push_back(pp.down());
  }
}

template <typename Dtype>
void PartitionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  CHECK(top.size() == left_.size()) << "Top blob' nums(" << top.size() 
      << ") isn't equal partition_param' nums(" << left_.size()<< ")";
  const int N = bottom[0]->num();
  const int C = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();

  for(int i = 0; i < left_.size(); ++i){
    CHECK(top[i]->cpu_data() != bottom[0]->cpu_data()) 
        << "PartitionLayer doesn't support inplace computing.";
    CHECK(left_[i] >= 0 && left_[i] < W) 
        << "left(" << left_[i] << ") in partition_param should be positive "
        << "and smaller than bottom blob's width(" << W << ")";
    CHECK(right_[i] >= 0 && right_[i] < W) 
        << "right(" << right_[i] << ") in partition_param should be positive "
        << "and smaller than bottom blob's width(" << W << ")";
    CHECK(up_[i] >= 0 && up_[i] < H)
        << "up(" << up_[i] << ") in partition_param should be positive "
        << "and smaller than bottom blob's height(" << H << ")";
    CHECK(down_[i] >= 0 && down_[i] < H)
        << "down(" << down_[i] << ") in partition_param should be positive "
        << "and smaller than bottom blob's height(" << H << ")";
    top[i]->Reshape(N, C, right_[i] - left_[i], down_[i] - up_[i]);
  }
}

template <typename Dtype>
void PartitionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data(); 
  const int N = bottom[0]->num();
  const int C = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
  int i, j, k, l;
  int tH, tW;
  Dtype* top_data;
  for (i = 0; i < left_.size(); ++i){
    top_data = top[i]->mutable_cpu_data();
    tH = top[i]->height();
    tW = top[i]->width();
    for (j = 0; j < N; ++j){
      for (k = 0; k < C; ++k){
        for (l = up_[i]; l < down_[i]; ++l){
          caffe_copy(right_[i] - left_[i], 
                     bottom_data + j*C*H*W + k*H*W + l*W + left_[i],
                     top_data + j*C*tH*tW + k*tH*tW + (l-up_[i])*tW);
        }
      }
    }
  }
}

template <typename Dtype>
void PartitionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  const Dtype* top_diff; 
  const int N = bottom[0]->num();
  const int C = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
  int i, j, k, l;
  int tH, tW;
  for (i = 0; i < left_.size(); ++i){
    top_diff = top[i]->cpu_diff();
    tH = top[i]->height();
    tW = top[i]->width();
    for (j = 0; j < N; ++j){
      for (k = 0; k < C; ++k){
        for (l = up_[i]; l < down_[i]; ++l){
          caffe_add(right_[i] - left_[i], 
                     bottom_diff + j*C*H*W + k*H*W + l*W + left_[i],
                     top_diff + j*C*tH*tW + k*tH*tW + (l-up_[i])*tW,
                     bottom_diff + j*C*H*W + k*H*W + l*W + left_[i]);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PartitionLayer);
#endif

INSTANTIATE_CLASS(PartitionLayer);
REGISTER_LAYER_CLASS(Partition);

} // namespace caffe
