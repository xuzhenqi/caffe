#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void TripletImageDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->StopInternalThread();
  // Reshape to loaded data.
  for (int i = 0; i < 3; ++i) {
    top[i]->Reshape(this->prefetch_data_[i]->num(),
                    this->prefetch_data_[i]->channels(),
                    this->prefetch_data_[i]->height(),
                    this->prefetch_data_[i]->width());
    // Copy the data
    caffe_copy(prefetch_data_[i]->count(), prefetch_data_[i]->cpu_data(),
               top[i]->mutable_gpu_data());
    DLOG(INFO) << "Prefetch copied";
  }
  // Start a new prefetch thread
  DLOG(INFO) << "StopInternalThread";
  this->StartInternalThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(TripletImageDataLayer);

}  // namespace caffe
