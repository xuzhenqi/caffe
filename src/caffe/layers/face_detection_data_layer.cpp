#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

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
  vector<int> label(10); //lex, ley, rex, rey, nx, ny, mlx, mly, mrx, mry;
  while (infile >> filename >> label[0] >> label[1] >> label[2] >> label[3]
      >> label[4] >> label[5] >> label[6] >> label[7] >> label[8] >> label[9]) {
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
  label_shape[1] = 10;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
  int height = top_shape[2], width = top_shape[3];
  CHECK_EQ(height%16, 0) << "height % 16 != 0";
  top[1]->Reshape(batch_size, 5, height, width);
  top[2]->Reshape(batch_size, 5, height/4, width/4);
  top[3]->Reshape(batch_size, 5, height/8, width/8);
  top[4]->Reshape(batch_size, 5, height/16, width/16);
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
  const int label_item_size = lines_[lines_id_].second.size();
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

    for (int i = 0; i < label_item_size; ++i) {
      prefetch_label[item_id*label_item_size + i] = lines_[lines_id_].second[i];
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
void FaceDetectionDataLayer<Dtype>::gauss_map(int x,
                                              int y,
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
void FaceDetectionDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue "
                                                  "empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";

  // Reshape to loaded labels.
  const int width = top[0]->width(), height = top[0]->height();
  const int num = top[0]->num();
  top[1]->Reshape(num, 5, height, width);
  top[2]->Reshape(num, 5, height/4, width/4);
  top[3]->Reshape(num, 5, height/8, width/8);
  top[4]->Reshape(num, 5, height/16, width/16);
  Dtype* label_data = batch->label_.mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < 5; ++j) {
      gauss_map(label_data[i*10+2*j], label_data[i*10+2*j+1], height, width,
                top[1]->mutable_cpu_data() + top[1]->offset(i, j));
    }
  }
  const Dtype* big_map = top[1]->cpu_data();
  Dtype* small_map = top[2]->mutable_cpu_data();
  for (int i = 0; i < num * 5; ++i) {
    for (int h = 0; h < height/4; ++h) {
      for (int w = 0; w < width/4; ++w) {
        *(small_map) = *(big_map);
        ++small_map;
        big_map += 4;
      }
      big_map += 3 * width;
    }
  }
  big_map = top[1]->cpu_data();
  small_map = top[3]->mutable_cpu_data();
  for (int i = 0; i < num * 5; ++i) {
    for (int h = 0; h < height/8; ++ h) {
      for (int w = 0; w < width/8; ++w) {
        *(small_map) = *(big_map);
        ++small_map;
        big_map += 8;
      }
      big_map += 7 * width;
    }
  }
  big_map = top[1]->cpu_data();
  small_map = top[4]->mutable_cpu_data();
  for (int i = 0; i < num * 5; ++i) {
    for (int h = 0; h < height/16; ++h) {
      for (int w = 0; w < width/16; ++w) {
        *(small_map) = *(big_map);
        ++small_map;
        big_map += 16;
      }
      big_map += 15 * width;
    }
  }

  this->prefetch_free_.push(batch);
}

INSTANTIATE_CLASS(FaceDetectionDataLayer);
REGISTER_LAYER_CLASS(FaceDetectionData);

}  // namespace caffe
#endif  // USE_OPENCV
