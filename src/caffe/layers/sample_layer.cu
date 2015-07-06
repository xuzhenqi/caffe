#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"



namespace caffe {

template <typename Dtype>
__global__ void sample_kernel(const int n, const int num, const Dtype *x, 
          Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = x[num * index];
  }
}

template <typename Dtype>
void SampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  sample_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, num_, bottom_data, top_data);
}

template <typename Dtype>
void SampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Sample layer need not backward.";
}

INSTANTIATE_LAYER_GPU_FUNCS(SampleLayer);

} // namespace caffe
