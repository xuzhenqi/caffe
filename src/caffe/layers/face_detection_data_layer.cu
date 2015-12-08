#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void FaceDetectionDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  /*
  Batch<Dtype>* batch = 
      this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to load data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  DLOG(INFO) << "Prefetch copied";

  // Reshape to load labels.
  const int width = top[0]->width(), height = top[0]->height();
  const int num = top[0]->num();
  top[1]->Reshape(num, 5, height, width);
  top[2]->Reshape(num, 5, height/4, width/4);
  top[3]->Reshape(num, 5, height/8, width/8);
  top[4]->Reshape(num, 5, height/16, width/16);
  Dtype* label_data = batch->label_.mutable_cpu_data();
*/
  // Test the performance and then decide whether to implement the gpu version.
  Forward_cpu(bottom, top);
}

INSTANTIATE_LAYER_GPU_FORWARD(FaceDetectionDataLayer);

} // namespace caffe
