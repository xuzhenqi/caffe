#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CombineLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  const int N = top[0]->num();
  const int C = top[0]->channels();
  const int H = top[0]->height();
  const int W = top[0]->width();
  int i, j, k, l;
  int tH, tW;
  const Dtype* bottom_data;
  for (i = 0; i < left_.size(); ++i){
    bottom_data = bottom[i]->gpu_data();
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
void CombineLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff; 
  const int N = top[0]->num();
  const int C = top[0]->channels();
  const int H = top[0]->height();
  const int W = top[0]->width();
  int i, j, k, l;
  int tH, tW;
  for (i = 0; i < left_.size(); ++i){
    bottom_diff = top[i]->mutable_gpu_diff();
    tH = top[i]->height();
    tW = top[i]->width();
    for (j = 0; j < N; ++j){
      for (k = 0; k < C; ++k){
        for (l = up_[i]; l < down_[i]; ++l){
          caffe_copy(right_[i] - left_[i], 
                     top_diff + j*C*H*W + k*H*W + l*W + left_[i]);
                     bottom_diff + j*C*tH*tW + k*tH*tW + (l-up_[i])*tW,
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CombineLayer);

} // namespace caffe
