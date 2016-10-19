#include <fstream>

#include "caffe/data_layers.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
SimpleDataLayer<Dtype>::~SimpleDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void SimpleDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  shuffle_ = this->layer_param_.simple_data_param().shuffle(); 
//  fields_begin_ = this->layer_param_.simple_data_param().fields_begin(); 
//  fields_end_ = this->layer_param_.simple_data_param().fields_end(); 
//  stride_ = this->layer_param_.simple_data_param().stride(); 
//  if (fields_end_ == -1) fields_end_ = stride_;
//  CHECK(fields_end_ > fields_begin_);
  batch_size_ = this->layer_param_.simple_data_param().batch_size();
  nums_ = this->layer_param_.simple_data_param().nums();
  const string filename = this->layer_param_.simple_data_param().filename(); 
  std::ifstream infile(filename.c_str());
  CHECK(infile) << "Open file failed: " << filename;
  string temp;
  while(infile >> temp) {
    buffer_.push_back(vector<Dtype>(nums_));
    for (int i = 0; i < nums_; ++i) infile >> buffer_.back()[i];
  }
  LOG(INFO) << "Loading " << buffer_.size() << " recods.";
  
  if (shuffle_) ShuffleData();

  vector<int> shape;
  shape.push_back(batch_size_);
  shape.push_back(nums_);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(shape);
  }
  top[0]->Reshape(shape);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels();
}

template <typename Dtype>
void SimpleDataLayer<Dtype>::ShuffleData() {
  shuffle(buffer_.begin(), buffer_.end());
}

template <typename Dtype>
void SimpleDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  vector<int> shape;
  shape.push_back(batch_size_);
  shape.push_back(nums_);
  batch->data_.Reshape(shape);
  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  for (int i = 0; i < batch_size_; ++i) {
    memcpy(prefetch_data + i * nums_, buffer_[cur_++].data(),
           sizeof(Dtype) * nums_); 
    if (cur_ == buffer_.size()) {
      cur_ = 0;
      if (shuffle_) ShuffleData();
    }
  }
}

INSTANTIATE_CLASS(SimpleDataLayer);
REGISTER_LAYER_CLASS(SimpleData);

}  // namespace caffe
