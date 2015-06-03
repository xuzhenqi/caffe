#include<vector>
#include<algorithm>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CombineLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  const int size = this->layer_param_.combine_param_size();
  CHECK(size > 1) << "Please give at least two combine_param.";
  left_.clear();
  right_.clear();
  up_.clear();
  down_.clear();
  for(int i=0; i<size; ++i){
    const PartitionParameter &pp = this->layer_param_.combine_param(i);
    left_.push_back(pp.left());
    right_.push_back(pp.right());
    up_.push_back(pp.up());
    down_.push_back(pp.down());
  }
}

template <typename Dtype>
void CombineLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  CHECK(bottom.size() == left_.size()) << "Bop blob' nums(" << top.size() 
      << ") isn't equal combine_param' nums(" << left_.size()<< ")";
  const int N = bottom[0]->num();
  const int C = bottom[0]->channels();

  int max_down = 0, max_right = 0, bN, bC, bH, bW;

  for(int i = 0; i < left_.size(); ++i){
    bN = bottom[i]->num();
    bC = bottom[i]->channels();
    bH = bottom[i]->height();
    bW = bottom[i]->width();
    CHECK(bN == N) << "Bottom blobs' dimension disagree. [num][0] " << N
        << " [num][" << i << "] " << bN;
    CHECK(bC == C) << "Bottom blobs' dimension disagree. [channels][0] " << C
        << " [channels][" << i << "] " << bC;
    CHECK(bH == down_[i] - up_[i]) << "Dimension Error";
    CHECK(bW == right_[i] - left_[i]) << "Dimension Error";
    max_down = std::max(max_down, down_[i]);
    max_right = std::max(max_right, right_[i]);
  }
  top[0]->Reshape(N, C, max_down, max_right);
}

template <typename Dtype>
void CombineLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  const int N = top[0]->num();
  const int C = top[0]->channels();
  const int H = top[0]->height();
  const int W = top[0]->width();
  int i, j, k, l;
  int tH, tW;
  const Dtype* bottom_data;
  for (i = 0; i < left_.size(); ++i){
    bottom_data = bottom[i]->cpu_data();
    tH = bottom[i]->height();
    tW = bottom[i]->width();
    for (j = 0; j < N; ++j){
      for (k = 0; k < C; ++k){
        for (l = up_[i]; l < down_[i]; ++l){
          caffe_copy(right_[i] - left_[i], 
                     bottom_data + j*C*tH*tW + k*tH*tW + (l-up_[i])*tW,
                     top_data + j*C*H*W + k*H*W + l*W + left_[i]);
        }
      }
    }
  }
}

template <typename Dtype>
void CombineLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff; 
  const int N = top[0]->num();
  const int C = top[0]->channels();
  const int H = top[0]->height();
  const int W = top[0]->width();
  int i, j, k, l;
  int tH, tW;
  for (i = 0; i < left_.size(); ++i){
    bottom_diff = top[i]->mutable_cpu_diff();
    tH = top[i]->height();
    tW = top[i]->width();
    for (j = 0; j < N; ++j){
      for (k = 0; k < C; ++k){
        for (l = up_[i]; l < down_[i]; ++l){
          caffe_copy(right_[i] - left_[i], 
                     top_diff + j*C*H*W + k*H*W + l*W + left_[i],
                     bottom_diff + j*C*tH*tW + k*tH*tW + (l-up_[i])*tW);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CombineLayer);
#endif

INSTANTIATE_CLASS(CombineLayer);
REGISTER_LAYER_CLASS(Combine);

} // namespace caffe
