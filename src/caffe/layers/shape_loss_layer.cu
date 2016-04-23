#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void interval_copy(const int N, const Dtype *x, const Dtype *y,
                              Dtype *z) {
  CUDA_KERNEL_LOOP(index, N) {
    z[2*index] = x[index];
    z[2*index+1] = y[index];
  }
}

template <typename Dtype>
void ShapeLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  Dtype* label = bottom[1]->mutable_gpu_data();
  const Dtype* row = multiplier_.gpu_data();
  const Dtype* col = multiplier_.gpu_diff();
  Dtype* mean_row = mean_shape_.mutable_gpu_data();
  Dtype* mean_col = mean_shape_.mutable_gpu_diff();
  Dtype* temp = bottom[1]->mutable_gpu_diff();
  int dim = prob_.count() / outer_num_;
  caffe_gpu_scal(bottom[1]->count(), Dtype(1.)/scale_, label);
  caffe_gpu_gemv(CblasNoTrans, outer_num_, dim, Dtype(1.), prob_data, row,
                 Dtype(0.), mean_row);
  caffe_gpu_gemv(CblasNoTrans, outer_num_, dim, Dtype(1.), prob_data, col,
                 Dtype(0.), mean_col);
  interval_copy<Dtype><<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
      dim, mean_col, mean_row, temp);
  caffe_gpu_sub(2*outer_num_, temp, label, temp);
  caffe_gpu_dot(2*outer_num_, temp, temp, top[0]->mutable_cpu_data());
  top[0]->mutable_cpu_data()[0] /= outer_num_;
  caffe_gpu_scal(bottom[1]->count(), scale_, label); // Rescale
}

template <typename Dtype>
__global__ void backward_kernel(const int N, const int H, const int W,
                                const Dtype loss_weight,
                                const Dtype *row, const Dtype *col, 
                                const Dtype *gap, const Dtype *prob, 
                                Dtype *diff) {
  CUDA_KERNEL_LOOP(index, N) {
    int k = index % W;
    int i = index / W;
    int j = i % H;
    i /= H;
    diff[index] = loss_weight * prob[index] * (gap[2*i] * (k - col[i]) + 
                  gap[2*i+1] * (j - row[i]));
  }
}

template <typename Dtype>
void ShapeLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                         const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* mean_row = mean_shape_.gpu_data();
  const Dtype* mean_col = mean_shape_.gpu_diff();
  const Dtype* label = bottom[1]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype loss_weight = top[0]->cpu_diff()[0] / outer_num_ * 2;
  int N = bottom[0]->count();
  backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, height_, width_, loss_weight, mean_row, 
      mean_col, bottom[1]->gpu_diff(), prob_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(ShapeLossLayer);

}  // namespace caffe
