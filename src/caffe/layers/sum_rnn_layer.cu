#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SumForward(const int n, const int dim, const Dtype* bottom_data, 
                           const Dtype* bottom_data_rnn,
                           const Dtype* begin_marker,
                           const Dtype coeff_data,
                           const Dtype coeff_data_rnn, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
    if (begin_marker[index / dim] <= 0.5) {
      top_data[index] = coeff_data * bottom_data[index] 
          + coeff_data_rnn * bottom_data_rnn[index];
    } else {
      top_data[index] = bottom_data[index];
    }
  }
}

template <typename Dtype>
__global__ void SumBackward(const int n, const int dim, const Dtype* top_diff,
                            const Dtype* begin_marker, 
                            const Dtype coeff_data, const Dtype coeff_data_rnn,
                            Dtype* bottom_diff, Dtype* bottom_diff_rnn) {
  CUDA_KERNEL_LOOP(index, n) {
    if (begin_marker[index / dim] <= 0.5) {
      bottom_diff[index] = coeff_data * top_diff[index];
      bottom_diff_rnn[index] = coeff_data_rnn * top_diff[index];
    } else {
      bottom_diff[index] = top_diff[index];
      bottom_diff_rnn[index] = 0;
    }
  }
}

template <typename Dtype>
void SumRNNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*> & bottom,
    const vector<Blob<Dtype>*> & top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_data_rnn = bottom[1]->gpu_data();
  const Dtype* begin_marker = bottom[2]->gpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = count / num;
  Dtype* top_data = top[0]->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SumForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, dim, bottom_data, bottom_data_rnn, begin_marker, 
      coeffs_[0], coeffs_[1], top_data);
}

template <typename Dtype>
void SumRNNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* begin_marker = bottom[2]->gpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = count / num;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff_rnn = bottom[1]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SumBackward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, dim, top_diff, begin_marker, coeffs_[0], coeffs_[1], 
      bottom_diff, bottom_diff_rnn);
}

INSTANTIATE_LAYER_GPU_FUNCS(SumRNNLayer);

} // namespace caffe
