#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PerturbationForward(const int n, const Dtype* in, 
    const Dtype* mask, const Dtype scale, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      out[index] = in[index] * mask[index] * scale;
    }
}

template <typename Dtype>
void PerturbationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    Dtype* mask = static_cast<Dtype*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_gaussian(count, mean_, std_, mask);
    PerturbationForward<Dtype><<<CAFFE_GET_BLOCKS(count), 
        CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, mask, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void PerturbationBackward(const int n, const Dtype* in_diff, 
    const Dtype* mask, const Dtype scale, Dtype* out_diff) {
    CUDA_KERNEL_LOOP(index, n) {
      out_diff[index] = in_diff[index] * mask[index] * scale;
    }
}

template <typename Dtype>
void PerturbationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const Dtype* mask = static_cast<const Dtype*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      PerturbationBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, scale_, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PerturbationLayer);

} // namespace caffe
