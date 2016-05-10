#include <vector>
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SumNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
  for(int i = 0; i < ranges_.size(); ++i) {
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int j = 0; j < ranges_[i].size(); ++j) {
        caffe_gpu_add(bottom[0]->count(2), top_data + top[0]->offset(n, i),
                  bottom_data + bottom[0]->offset(n, ranges_[i][j]),
                  top_data + top[0]->offset(n, i));
      }
      caffe_gpu_scal(top[0]->count(2), Dtype(1. / ranges_[i].size()), 
                     top_data + top[0]->offset(n, i));
    }
  }
}

template <typename Dtype>
void SumNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                       const vector<bool> &propagate_down,
                                       const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
    for (int i = 0; i < ranges_.size(); ++i) {
      for (int n = 0; n < bottom[0]->num(); ++n) {
        for (int j = 0; j < ranges_[i].size(); ++j) {
          caffe_gpu_axpy(top[0]->count(2), Dtype(1./ranges_[i].size()),
                     top_diff + top[0]->offset(n, i),
                     bottom_diff + bottom[0]->offset(n, ranges_[i][j]));
        }
      }
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SumNormLayer);

} // namespace caffe
