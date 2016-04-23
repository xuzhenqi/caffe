#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template<typename Dtype>
__global__ void gauss_map_kernel(const int n, const Dtype *label_data, 
                                 const int height, 
                                 const int width, const Dtype std, 
                                 Dtype *data) {
  CUDA_KERNEL_LOOP(index, n) {
    int w = index % width;
    int p = index / width;
    int h = p % height;
    p /= height;
    Dtype w_gap = w - label_data[2*p];
    Dtype h_gap = h - label_data[2*p + 1];
    data[index] = expf((-w_gap*w_gap-h_gap*h_gap)/std/std);
  }
}

template<typename Dtype>
__global__ void down_sample_kernel(const int n, const int scale, 
                                   const int height, const int width,
                                   const Dtype *big_map, Dtype *small_map) {
  CUDA_KERNEL_LOOP(index, n) {
    int w = index % width * scale;
    int p = index / width;
    small_map[index] = big_map[p * scale * width * scale + w];
  }
}

template <typename Dtype>
void FaceDetectionDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue "
                                                  "empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  top[1]->ReshapeLike(batch->label_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
             top[0]->mutable_gpu_data());
  caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
             top[1]->mutable_gpu_data());
  DLOG(INFO) << "Prefetch copied";

  // Reshape to loaded labels.
  const int width = top[0]->width(), height = top[0]->height();
  const int num = top[0]->num(), count = top[2]->count();
  top[2]->Reshape(num, points_, height, width);
  for (int i = 0; i < scales_.size(); ++i)
    top[i+3]->Reshape(num, points_, height/scales_[i], width/scales_[i]);

  const Dtype* label_data = batch->label_.gpu_data();
  gauss_map_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, label_data, height, width, std_, 
      top[2]->mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num*points_, height*width, 1., 
                        top[2]->gpu_data(), sum_multiplier_.gpu_data(), 0., 
                        sum_multiplier_.mutable_gpu_diff());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num*points_, height*width, 
                        1., 1., sum_multiplier_.gpu_diff(), 
                        sum_multiplier_.gpu_data(), 0., 
                        top[2]->mutable_gpu_diff());
  caffe_gpu_div<Dtype>(top[2]->count(), top[2]->gpu_data(), top[2]->gpu_diff(),
                       top[2]->mutable_gpu_data());
  int sub_count, sub_height, sub_width;
  for (int s = 0; s < scales_.size(); ++s) {
    sub_count = count / scales_[s] / scales_[s];
    sub_height = height / scales_[s];
    sub_width = width / scales_[s];
    down_sample_kernel<Dtype><<<CAFFE_GET_BLOCKS(sub_count), 
        CAFFE_CUDA_NUM_THREADS>>>(sub_count, scales_[s], 
         sub_height, sub_width, top[2]->gpu_data(),
         top[3+s]->mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num*points_, sub_height*sub_width, 1., 
                          top[3+s]->gpu_data(), sum_multiplier_.gpu_data(), 0., 
                          sum_multiplier_.mutable_gpu_diff());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num*points_, 
                          sub_height*sub_width, 1., 1., 
                          sum_multiplier_.gpu_diff(), 
                          sum_multiplier_.gpu_data(), 0., 
                          top[3+s]->mutable_gpu_diff());
    caffe_gpu_div<Dtype>(sub_count, top[3+s]->gpu_data(), top[3+s]->gpu_diff(),
                         top[3+s]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  this->prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(FaceDetectionDataLayer);

} // namespace caffe
