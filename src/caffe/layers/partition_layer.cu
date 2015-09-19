#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void block_copy_kernel(const int N, const int sH, const int sW, 
      const int dH, const int dW, const int sup, const int sleft, 
      const int dup, const int dleft, const int tH, const int tW, 
      const Dtype* src, Dtype* dst) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < N && y < tH && z < tW) {
    dst[x*dH*dW + (y+dup)*dW + z+dleft] = src[x*sH*sW + (y+sup)*sW + z+sleft];
  }
}

template <typename Dtype>
__global__ void block_add_kernel(const int N, const int sH, const int sW, 
      const int dH, const int dW, const int sup, const int sleft, 
      const int dup, const int dleft, const int tH, const int tW, 
      const Dtype* src, Dtype* dst) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < N && y < tH && z < tW) {
    dst[x*dH*dW + (y+dup)*dW + z+dleft] += src[x*sH*sW + (y+sup)*sW + z+sleft];
  }
}

template <typename Dtype>
void PartitionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data(); 
  const int N = bottom[0]->num();
  const int C = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
  int i, j, k, l;
  int tH, tW;
  Dtype* top_data;
  for (i = 0; i < left_.size(); ++i){
    top_data = top[i]->mutable_gpu_data();
    tH = top[i]->height();
    tW = top[i]->width();
    dim3 dimBlock(CAFFE_CUDA_NUM_THREADS/256, 16, 16);
    dim3 dimGrid((N*C + dimBlock.x - 1)/dimBlock.x, 
                 (tH + dimBlock.y - 1)/dimBlock.y,
                 (tW + dimBlock.z - 1)/dimBlock.z);
    block_copy_kernel<Dtype><<<dimGrid, dimBlock>>>(N*C, H, W, tH, tW, up_[i], 
                  left_[i], 0, 0, tH, tW, bottom_data, top_data);
/*
    for (j = 0; j < N; ++j){
      for (k = 0; k < C; ++k){
        for (l = up_[i]; l < down_[i]; ++l){
          caffe_copy(right_[i] - left_[i], 
                     bottom_data + j*C*H*W + k*H*W + l*W + left_[i],
                     top_data + j*C*tH*tW + k*tH*tW + (l-up_[i])*tW);
        }
      }
    }
*/
  }
}

template <typename Dtype>
void PartitionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  const Dtype* top_diff; 
  const int N = bottom[0]->num();
  const int C = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
  int i;
  int tH, tW;
  for (i = 0; i < left_.size(); ++i){
    top_diff = top[i]->gpu_diff();
    tH = top[i]->height();
    tW = top[i]->width();
    dim3 dimBlock(CAFFE_CUDA_NUM_THREADS/256, 16, 16);
    dim3 dimGrid((N*C + dimBlock.x - 1)/dimBlock.x, 
                 (tH + dimBlock.y - 1)/dimBlock.y,
                 (tW + dimBlock.z - 1)/dimBlock.z);
    block_add_kernel<Dtype><<<dimGrid, dimBlock>>>(N*C, tH, tW, H, W, 0, 0, up_[i],
            left_[i], tH, tW, top_diff, bottom_diff);
/*
    for (j = 0; j < N; ++j){
      for (k = 0; k < C; ++k){
        for (l = up_[i]; l < down_[i]; ++l){
          caffe_gpu_add(right_[i] - left_[i], 
                     bottom_diff + j*C*H*W + k*H*W + l*W + left_[i],
                     top_diff + j*C*tH*tW + k*tH*tW + (l-up_[i])*tW,
                     bottom_diff + j*C*H*W + k*H*W + l*W + left_[i]);
        }
      }
    }
*/
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PartitionLayer);

} // namespace caffe
