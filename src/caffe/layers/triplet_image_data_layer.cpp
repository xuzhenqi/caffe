#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
  
template <typename Dtype>
TripletImageDataLayer<Dtype>::~TripletImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  delete this->prefetch_data_[0];
  delete this->prefetch_data_[1];
  delete this->prefetch_data_[2];
}

template <typename Dtype>
void TripletImageDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*> &bottom,
    const vector<Blob<Dtype>*> &top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make three
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_[0]->mutable_cpu_data();
  this->prefetch_data_[1]->mutable_cpu_data();
  this->prefetch_data_[2]->mutable_cpu_data();

  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void TripletImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->output_labels_ = false;

  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the statics
  const string &statics = 
      this->layer_param_.triplet_image_data_param().statics();
  LOG(INFO) << "Opening file " << statics;
  std::ifstream infile(statics.c_str());
  string filename;
  int nums, label, size = 0;
  infile >> filename >> filename >> filename; // ignore the first line
  while (infile >> filename >> nums >> label) {
    CHECK_EQ(label, label_size_.size()) 
        << "label should start from 0 to n continuously";
    label_size_.push_back(size);
    size += nums;
  }
  label_size_.push_back(size);
  infile.close();
  
  // Read the file with filenames and labels
  lines_.resize(size);
  labels_.resize(size);
  const string &source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  infile.open(source.c_str());
  vector<int> label_to(label_size_);
  while (infile >> filename >> label) {
    CHECK(lines_[label_to[label]].empty());
    labels_[label_to[label]] = label;
    lines_[label_to[label]++] = filename;
  }
  
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    LOG(FATAL) << "rand_skip is not supported in TripletImageDataLayer";
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[0],
                                    new_height, new_width, is_color);
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  this->prefetch_data_.push_back(new Blob<Dtype>());
  this->prefetch_data_.push_back(new Blob<Dtype>());
  this->prefetch_data_.push_back(new Blob<Dtype>());
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    top[1]->Reshape(batch_size, channels, crop_size, crop_size);
    top[2]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_data_[0]->Reshape(
        batch_size, channels, crop_size, crop_size);
    this->prefetch_data_[1]->Reshape(
        batch_size, channels, crop_size, crop_size);
    this->prefetch_data_[2]->Reshape(
        batch_size, channels, crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    top[1]->Reshape(batch_size, channels, height, width);
    top[2]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_[0]->Reshape(
        batch_size, channels, height, width);
    this->prefetch_data_[1]->Reshape(
        batch_size, channels, height, width);
    this->prefetch_data_[2]->Reshape(
        batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}

template <typename Dtype>
void TripletImageDataLayer<Dtype>::GetIndexs(vector<vector<int> > &indexs) {
  int batch_size = indexs.size();
  int size = lines_.size(), temp, temp2;
  CHECK_EQ(3, indexs[0].size());
  for (int i = 0; i < batch_size; ++i) {
    // Pick the anchor
    indexs[i][0] = caffe_rng_rand() % size;
    // Pick the positive
    temp = labels_[indexs[i][0]];
    temp2 = label_size_[temp + 1] - label_size_[temp];
    indexs[i][1] = caffe_rng_rand() % (temp2 - 1) + label_size_[temp];
    if (indexs[i][1] >= indexs[i][0])
      ++indexs[i][1];
    // Pick the negative
    indexs[i][2] = caffe_rng_rand() % (size - temp2);
    if (indexs[i][2] >= label_size_[temp])
      indexs[i][2] += temp2;
  }
}


template <typename Dtype>
void TripletImageDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_[0]->count());
  CHECK(this->prefetch_data_[1]->count());
  CHECK(this->prefetch_data_[2]->count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();
  
  vector<vector<int> > indexs(batch_size, vector<int>(3, 0));
  GetIndexs(indexs);
  for (int i = 0; i < batch_size; ++i) {
    // get a blob
    timer.Start();
    for (int j = 0; j < 3; ++j) {
      //std::cout << "index: " << indexs[i][j] 
      //    << "\t filename: " << lines_[indexs[i][j]] << std::endl;
      cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[indexs[i][j]],
                      new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[indexs[i][j]];
      read_time += timer.MicroSeconds();
      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      int offset = this->prefetch_data_[j]->offset(i);
      this->transformed_data_.set_cpu_data(
          prefetch_data_[j]->mutable_cpu_data() + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      trans_time += timer.MicroSeconds();
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void TripletImageDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Reshape to loaded data.
  for (int i = 0; i < 3; ++i) {
    top[i]->Reshape(this->prefetch_data_[i]->num(),
                    this->prefetch_data_[i]->channels(),
                    this->prefetch_data_[i]->height(),
                    this->prefetch_data_[i]->width());
    // Copy the data
    caffe_copy(prefetch_data_[i]->count(), prefetch_data_[i]->cpu_data(),
               top[i]->mutable_cpu_data());
    DLOG(INFO) << "Prefetch copied";
  }
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  this->CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(TripletImageDataLayer, Forward);
#endif

INSTANTIATE_CLASS(TripletImageDataLayer);
REGISTER_LAYER_CLASS(TripletImageData);

} // namespace caffe
