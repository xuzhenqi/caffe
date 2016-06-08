#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
FaceDetectionDataLayer<Dtype>::~FaceDetectionDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void FaceDetectionDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std_ = this->layer_param_.gaussmap_param().std();
  points_ = this->layer_param_.gaussmap_param().points();
  scales_.clear();
  for (int i = 0; i < this->layer_param_.gaussmap_param().scale_size(); ++i) {
    scales_.push_back(this->layer_param_.gaussmap_param().scale(i));
  }
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  vector<Dtype> label(2 * points_);
  while (infile >> filename) {
    for (int i = 0; i < 2 * points_; ++i) {
      infile >> label[i];
    }
    lines_.push_back(std::make_pair(filename, label));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(2, 0);
  label_shape[0] = batch_size;
  label_shape[1] = 2 * points_;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
  int height = top_shape[2], width = top_shape[3];

  top[2]->Reshape(batch_size, points_, height, width);
  CHECK_EQ(scales_.size(), top.size() - 3);
  for (int i = 0; i < scales_.size(); ++i) {
    CHECK_EQ(height % scales_[i], 0);
    CHECK_EQ(width % scales_[i], 0);
    top[i+3]->Reshape(batch_size, points_, height/scales_[i], width/scales_[i]);
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    sum_multiplier_.Reshape(top[0]->shape());
    caffe_gpu_set(top[0]->count(), Dtype(1.), sum_multiplier_
        .mutable_gpu_data());
  }
#endif
}

template <typename Dtype>
void FaceDetectionDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void FaceDetectionDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    for (int i = 0; i < 2 * points_;  ++i) {
      prefetch_label[item_id *2 * points_ + i] = lines_[lines_id_].second[i];
    }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void FaceDetectionDataLayer<Dtype>::gauss_map(float x,
                                              float y,
                                              int height,
                                              int width,
                                              Dtype *map) {
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      map[h*width + w] = expf((-(h-y)*(h-y)-(w-x)*(w-x))/std_/std_);
    }
  }
}

template <typename Dtype>
void FaceDetectionDataLayer<Dtype>::regularize(int num, Dtype *map) {
  Dtype sum = 0;
  for (int i = 0; i < num; ++i) {
    sum += map[i];
  }
  for (int i = 0; i < num; ++i) {
    map[i] /= sum;
  }
}

template <typename Dtype>
void FaceDetectionDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue "
                                                  "empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  top[1]->ReshapeLike(batch->label_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
            top[1]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";

  // Reshape to loaded labels.
  const int width = top[0]->width(), height = top[0]->height();
  const int num = top[0]->num();
  top[2]->Reshape(num, points_, height, width);
  for (int i = 0; i < scales_.size(); ++i)
    top[i+3]->Reshape(num, points_, height/scales_[i], width/scales_[i]);

  Timer timer;
  double gauss_time = 0.0, map_time = 0.0;
  timer.Start();
  Dtype* label_data = batch->label_.mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < points_; ++j) {
      gauss_map(label_data[i*2*points_+2*j], label_data[i*2*points_+2*j+1],
                height, width,
                top[2]->mutable_cpu_data() + top[2]->offset(i, j));
      regularize(height * width,
                 top[2]->mutable_cpu_data() + top[2]->offset(i, j));
    }
  }
  gauss_time += timer.MicroSeconds();
  timer.Start();
  const Dtype* big_map;
  Dtype *small_map = NULL;
  for (int s = 0; s < scales_.size(); ++s) {
    big_map = top[2]->cpu_data();
    small_map = top[s+3]->mutable_cpu_data();
    for (int i = 0; i < num * points_; ++i) {
      for (int h = 0; h < height / scales_[s]; ++h) {
        for (int w = 0; w < width / scales_[s]; ++w) {
          *small_map = *big_map;
          ++small_map;
          big_map += scales_[s];
        }
        big_map += (scales_[s]-1) * width;
      }
      regularize(height / scales_[s] * width / scales_[s], small_map - height /
          scales_[s] * width / scales_[s]);
    }
  }
  map_time += timer.MicroSeconds();
  DLOG(INFO) << "Gauss time: " << gauss_time / 1000 << " ms.";
  DLOG(INFO) << "Map time: " << map_time / 1000 << " ms.";

  this->prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(FaceDetectionDataLayer, Forward);
#endif

INSTANTIATE_CLASS(FaceDetectionDataLayer);
REGISTER_LAYER_CLASS(FaceDetectionData);

}  // namespace caffe
