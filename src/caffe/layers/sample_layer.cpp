#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void SampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = this->layer_param_.sample_param().num();
  CHECK(bottom[0]->count() % num_ == 0);
  CHECK(bottom[0]->channels() == 1);
  CHECK(bottom[0]->width() == 1);
  CHECK(bottom[0]->height() == 1);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void SampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num()/num_, 1, 1, 1);
}

template <typename Dtype>
void SampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  int j;
  for(int i = 0; i < count; ++i){
    top_data[i] = bottom_data[num_*i];
    for (j = 1; j < num_; ++j)
      CHECK(top_data[i] == bottom_data[num_*i + j]);
  }
}

template <typename Dtype>
void SampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "SampleLayer need not backward";
}

#ifdef CPU_ONLY
STUB_GPU(SampleLayer);
#endif

INSTANTIATE_CLASS(SampleLayer);
REGISTER_LAYER_CLASS(Sample);

}  // namespace caffe
