#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <sched.h>
#include <sys/sysinfo.h>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

namespace caffe {

template <typename Dtype>
void ExImageDataLayer<Dtype>::CreatePrefetchThread() {
  if(this->layer_param_.ex_image_data_param().debug_info()) {
    LOG(INFO) << "[Multithread] CreatePrefetchThread starting ...";
  }
  next_prefetch_thread_ = 0;
  this->data_transformer_->InitRand();
  for (int i = 0; i < prefetch_thread_num; ++i) {
    try {
      this->prefetch_thread_v_[i]->thread_.reset(
          new std::thread(&ExImageDataLayer::InternalThreadEntry, this, i));
    } catch (...) {
      CHECK(false) << "Thread create failed!";
    }
    shared_ptr<PrefetchThreadInfo<Dtype>>& cur_prefetch_info =
        this->prefetch_thread_v_[i];
    GetFetchList(cur_prefetch_info->list_);
    cur_prefetch_info->start_.signal();
    if(this->layer_param_.ex_image_data_param().debug_info()) {
      LOG(INFO) << "[Multithread] " << i << " signal()";
    }
  }
  if(this->layer_param_.ex_image_data_param().debug_info()) {
    LOG(INFO) << "[Multithread] CreatePrefetchThread completed.";
  }
}

template <typename Dtype>
void ExImageDataLayer<Dtype>::JoinPrefetchThread() {
  if(this->layer_param_.ex_image_data_param().debug_info()) {
    LOG(INFO) << "[Multithread] JoinPrefetchThread starting ...";
  }
  for (int i = 0; i < prefetch_thread_num; ++i) {
    shared_ptr<PrefetchThreadInfo<Dtype>>& cur_prefetch_info =
        this->prefetch_thread_v_[i];
    cur_prefetch_info->end_.wait();
    cur_prefetch_info->list_.clear();
    cur_prefetch_info->start_.signal();
    try {
      cur_prefetch_info->thread_->join();
    } catch (...) {
      CHECK(false) << "Thread " << i << " join failed!";
    }
  }
  if(this->layer_param_.ex_image_data_param().debug_info()) {
    LOG(INFO) << "[Multithread] JoinPrefetchThread end.";
  }
}

template <typename Dtype>
ExImageDataLayer<Dtype>::~ExImageDataLayer<Dtype>() {
    this->JoinPrefetchThread();
}

template <typename Dtype>
void ExImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {


const ExImageDataParameter &ex_image_data_param=this->layer_param_.ex_image_data_param();

this->transform_type_ = this->layer_param_.transform_type();
CHECK_NE(this->transform_type_, LayerParameter_TransformType_NATIVE)
<< "\n Not support native data transformer";
if(this->transform_type_ == LayerParameter_TransformType_AUGMENT) {
CHECK(this->layer_param_.has_aug_transform_param())
<< "\n AugDataTransformer need parameters";
}

binary_source=ex_image_data_param.binary_source();
for(size_t i=0;i<ex_image_data_param.source_size();i++) {
source.push_back(ex_image_data_param.source(i));
}
for(size_t i=0;i<ex_image_data_param.root_folder_size();i++) {
root_folder.push_back(ex_image_data_param.root_folder(i));
}
CHECK_GE(source.size(), 1)
<< "\n At least 1 source is needed";
CHECK(root_folder.size()==source.size() || root_folder.size()<=1)
<< "\n Each source file should have a root folder, or use the same root folder, or use no root folder";
if(root_folder.size()==0) {
root_folder.push_back("");
}
if(root_folder.size()==1) {
for(size_t i=1;i<source.size();i++) {
root_folder.push_back(root_folder[0]);
}
}

batch_size = ex_image_data_param.batch_size();
CHECK_GT(batch_size, 0)
<<"\n batch_size must be positive";

new_height = ex_image_data_param.new_height();
new_width  = ex_image_data_param.new_width();
CHECK((new_height == 0 && new_width == 0) || (new_height > 0 && new_width > 0))
<< "\n Current implementation requires new_height and new_width to be set at the same time";
new_max_size = ex_image_data_param.new_max_size();

data_type = ex_image_data_param.data_type();
data_num = ex_image_data_param.data_num();
CHECK_GT(data_num, 0)
<< "\n data num must be positive";
label_type = ex_image_data_param.label_type();
label_num = ex_image_data_param.label_num();
CHECK_GT(data_num, 0)
<< "\n label num must be positive";

need_shuffle = ex_image_data_param.shuffle();
is_color = ex_image_data_param.is_color();
cache_in_byte = ex_image_data_param.cache_in_gb()*(unsigned long long)1024*1024*1024;
used_cache_in_byte = 0;

// Read the list
for(int i=0;i<source.size();i++) {
LOG(INFO) << "Opening file " << source[i] << "  ...";
if(binary_source) {
// FILE *fp_in=fopen(source[i].c_str(), "rb");
// CHECK(fp_in)
// << "\n Open failed: " << source[p];
//
// size_t total_num=0;
// CHECK_EQ(fread(&total_num, sizeof(size_t), 1, fp_in), 1)
// << "\n Open failed: " << source[p];
// for(size_t i=0;i<total_num;i++) {
//     size_t str_len;
//     CHECK_EQ(fread(&str_len, sizeof(size_t), 1, fp_in), 1)
//     << "\n Open failed: " << source[p];
//     string filename;
//     filename.resize(str_len);
//     CHECK_EQ(fread(&(filename[0]), sizeof(char), str_len, fp_in), str_len)
//     << "\n Open failed: " << source[p];
//     vector<float> label_float(label_num);
//         CHECK_EQ(fread(&(label_float[0]), sizeof(float), label_num, fp_in), label_num)
//     << "\n Open failed: " << source[p];
//
//     vector<Dtype> label(label_num);
//     for(size_t i=0;i<label_num;i++) {
//         label[i]=label_float[i];
//     }
//     lines_.push_back(make_pair(root_folder[p]+filename, label));
// }
// fclose(fp_in);
NOT_IMPLEMENTED;
}
else {
std::ifstream infile;
infile.open(source[i].c_str(), std::ifstream::in);
CHECK(infile.is_open())
<< "\n Open failed: " << source[i];
for(;;) {
bool success=true;
vector<string> label;
if(label_type==ExImageDataParameter_DataType_IMAGE) {
for(size_t j=0;j<label_num;j++) {
string temp;
if(!(infile>>temp)) {
success=false;
break;
}
temp=temp.substr(0, temp.find_first_of("\r\n"));
label.push_back(root_folder[i]+temp);
}
}
else {
vector<Dtype> label_dtype;
for(size_t j=0;j<label_num;j++) {
float temp;
if(!(infile>>temp)) {
success=false;
break;
}
label_dtype.push_back(temp);
}
string one_label;
one_label.resize(label_dtype.size()*sizeof(Dtype));
memcpy(&one_label[0], label_dtype.data(), label_dtype.size()*sizeof(Dtype));
label.push_back(one_label);
}

vector<string> data;
if(data_type==ExImageDataParameter_DataType_IMAGE) {
// TODO: Jinwei: Some previous lists have path with "space"
// So we still read to line's end for data_type=IMAGE and data_num=1
if(data_num==1) {
infile.get();
string temp;
getline(infile, temp);
temp=temp.substr(0, temp.find_first_of("\r\n"));
if(temp.size()==0) {
success=false;
}
else {
data.push_back(root_folder[i]+temp);
}
}
else {
for(size_t j=0;j<data_num;j++) {
string temp;
if(!(infile>>temp)) {
success=false;
break;
}
temp=temp.substr(0, temp.find_first_of("\r\n"));
data.push_back(root_folder[i]+temp);
}
}
}
else {
vector<Dtype> data_dtype;
for(size_t j=0;j<data_num;j++) {
float temp;
if(!(infile>>temp)) {
success=false;
break;
}
data_dtype.push_back(temp);
}
string one_data;
one_data.resize(data_dtype.size()*sizeof(Dtype));
memcpy(&one_data[0], data_dtype.data(), data_dtype.size()*sizeof(Dtype));
data.push_back(one_data);
}

if(success) {
lines_.push_back(make_pair(data, label));
}
else {
break;
}
}
infile.close();
}
}
CHECK(lines_.size()>0)
<< "\n No images in this list";
lines_index.resize(lines_.size());
for(size_t i=0;i<lines_.size();i++) {
lines_index[i]=i;
}
lines_id_=0;

set<string> image_set;
for(size_t i=0;i<lines_.size();i++) {
if(data_type==ExImageDataParameter_DataType_IMAGE) {
for(size_t j=0;j<lines_[i].first.size();j++) {
image_set.insert(lines_[i].first[j]);
}
}
if(label_type==ExImageDataParameter_DataType_IMAGE) {
for(size_t j=0;j<lines_[i].second.size();j++) {
image_set.insert(lines_[i].second[j]);
}
}
}
total_image_num=image_set.size();
LOG(INFO) << "A total of " << total_image_num << " images.";

cache_finished=false;

// randomly shuffle data
if (need_shuffle) {
LOG(INFO) << "Shuffling data ...";
prefetch_rng_.reset(new Caffe::RNG(caffe_rng_rand()));
ShuffleImages();
}

prefetch_thread_num = ex_image_data_param.thread_num();

if(this->layer_param_.ex_image_data_param().debug_info()) {
LOG(INFO) << "[Multithread] prefetch_thread_num: " <<prefetch_thread_num;
}
this->prefetch_thread_v_.resize(prefetch_thread_num);
for (int ipt = 0; ipt < prefetch_thread_num; ++ipt) {
this->prefetch_thread_v_[ipt].reset(new PrefetchThreadInfo<Dtype>());
this->prefetch_thread_v_[ipt]->transformed_data_v.resize(batch_size);
this->prefetch_thread_v_[ipt]->transformed_label_v.resize(batch_size);
for (size_t i = 0; i < batch_size; i++) {
this->prefetch_thread_v_[ipt]->transformed_data_v[i].reset(
new Blob<Dtype>);
this->prefetch_thread_v_[ipt]->transformed_label_v[i].reset(
new Blob<Dtype>);
}
}
if(this->layer_param_.ex_image_data_param().debug_info()) {
LOG(INFO) << "[Multithread] Initialize prefetch_thread_v_";
}

// reshape blobs
cv::Mat cv_data;
cv::Mat cv_label;
LoadData(lines_[lines_index[lines_id_]].first, data_type, data_num, cv_data);
LoadData(lines_[lines_index[lines_id_]].second, label_type, label_num, cv_label);
if(this->transform_type_ == LayerParameter_TransformType_AUGMENT) {
this->aug_data_transformer_->Transform(
    cv_data, cv_label,
*(this->prefetch_thread_v_[0]->transformed_data_v[0]),
*(this->prefetch_thread_v_[0]->transformed_label_v[0]));
vector<int> data_shape=
    this->prefetch_thread_v_[0]->transformed_data_v[0]->shape();
vector<int> label_shape=
    this->prefetch_thread_v_[0]->transformed_label_v[0]->shape();
data_shape[0]=batch_size;
label_shape[0]=batch_size;
top[0]->Reshape(data_shape);
top[1]->Reshape(label_shape);
}
else {
if(data_type==ExImageDataParameter_DataType_IMAGE) {
top[0]->Reshape(batch_size, cv_data.channels(), cv_data.rows, cv_data.cols);
}
else {
vector<int> data_shape(2);
data_shape[0]=batch_size;
data_shape[1]=data_num;
top[0]->Reshape(data_shape);
}

if(label_type==ExImageDataParameter_DataType_IMAGE) {
top[1]->Reshape(batch_size, cv_label.channels(), cv_label.rows, cv_label.cols);
}
else {
vector<int> label_shape(2);
label_shape[0]=batch_size;
label_shape[1]=label_num;
top[1]->Reshape(label_shape);
}
}

}

// rewrite the LayerSetp to init prefetch in main process, check data transformer
template <typename Dtype>
void ExImageDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
if (top.size() == 1) {
this->output_labels_ = false;
} else {
this->output_labels_ = true;
}
CHECK_NE(this->layer_param_.transform_type(), LayerParameter_TransformType_NATIVE)
<< "\n Not support native data transformer";
this->data_transformer_.reset(new DataTransformer<Dtype>(this->layer_param_.transform_param(), this->phase_));
this->data_transformer_->InitRand();
if(this->layer_param_.transform_type()==LayerParameter_TransformType_AUGMENT) {
this->aug_data_transformer_.reset(new AugDataTransformer<Dtype>(this->layer_param_.aug_transform_param(), this->phase_));
this->aug_data_transformer_->InitRand();
}
DataLayerSetUp(bottom, top);

this->CreatePrefetchThread();
}

// rewrite the forward to apply forward only in main process
template <typename Dtype>
void ExImageDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
shared_ptr<PrefetchThreadInfo<Dtype>> cur_thread_info =
    prefetch_thread_v_[next_prefetch_thread_++];
if (next_prefetch_thread_ == prefetch_thread_num) {
next_prefetch_thread_ = 0;
}

cur_thread_info->end_.wait();
GetFetchList(cur_thread_info->list_);
cur_thread_info->start_.signal();

// Reshape to loaded data.
top[0]->ReshapeLike(cur_thread_info->data_);
// Copy the data
caffe_copy(cur_thread_info->data_.count(),
    cur_thread_info->data_.cpu_data(),
    top[0]->mutable_cpu_data());
DLOG(INFO) << "Prefetch copied";
if (this->output_labels_) {
// Reshape to loaded labels.
top[1]->ReshapeLike(cur_thread_info->label_);
// Copy the labels.
caffe_copy(cur_thread_info->label_.count(),
    cur_thread_info->label_.cpu_data(),
    top[1]->mutable_cpu_data());
}
}

template <typename Dtype>
void ExImageDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
shared_ptr<PrefetchThreadInfo<Dtype>> cur_thread_info =
    prefetch_thread_v_[next_prefetch_thread_++];
if (next_prefetch_thread_ == prefetch_thread_num) {
next_prefetch_thread_ = 0;
}

cur_thread_info->end_.wait();
GetFetchList(cur_thread_info->list_);
cur_thread_info->start_.signal();

// Reshape to loaded data.
top[0]->ReshapeLike(cur_thread_info->data_);
// Copy the data
caffe_copy(cur_thread_info->data_.count(),
    cur_thread_info->data_.gpu_data(),
    top[0]->mutable_gpu_data());
DLOG(INFO) << "Prefetch copied";
if (this->output_labels_) {
// Reshape to loaded labels.
top[1]->ReshapeLike(cur_thread_info->label_);
// Copy the labels.
caffe_copy(cur_thread_info->label_.count(),
    cur_thread_info->label_.gpu_data(),
    top[1]->mutable_gpu_data());
}
}

// to shuffle images
template <typename Dtype>
void ExImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_index.begin(), lines_index.end(), prefetch_rng);
}

template <typename Dtype>
void ExImageDataLayer<Dtype>::GetFetchList(vector<size_t>& list) {
  list.resize(batch_size);
  for(size_t i=0;i<batch_size;i++) {
    list[i]=lines_index[lines_id_];
    lines_id_++;
    if (lines_id_ >= lines_.size()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (need_shuffle) {
        DLOG(INFO) << "Shuffling data ...";
        ShuffleImages();
      }
    }
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ExImageDataLayer<Dtype>::InternalThreadEntry(int index) {
  cpu_set_t cpu_mask;
  CPU_ZERO(&cpu_mask);
  int ncpus=get_nprocs();
  for(size_t i=0;i<ncpus;i++) {
    CPU_SET(i, &cpu_mask);
  }
  sched_setaffinity(0, sizeof(cpu_mask), &cpu_mask);

  shared_ptr<PrefetchThreadInfo<Dtype>>& cur_thread_info=this->prefetch_thread_v_[index];
  while(true) {
    if(this->layer_param_.ex_image_data_param().debug_info()) {
      LOG(INFO) << "thread ID " << index <<  " wait()";
    }
    cur_thread_info->start_.wait();
    if (cur_thread_info->list_.empty()) {
      if(this->layer_param_.ex_image_data_param().debug_info()) {
        LOG(INFO) << "thread ID " << index <<  " exit";
      }
      break;
    }

    if(this->layer_param_.ex_image_data_param().debug_info()) {
      LOG(INFO) << "thread ID " << index <<  " start loading data";
    }
    for (int i = 0; i < batch_size; i++) {
      cv::Mat cv_data;
      cv::Mat cv_label;
      LoadData(lines_[cur_thread_info->list_[i]].first, data_type,
               data_num, cv_data);
      LoadData(lines_[cur_thread_info->list_[i]].second, label_type,
               label_num, cv_label);
      if (this->transform_type_ == LayerParameter_TransformType_NONE) {
        MatToBlob(cv_data, *(cur_thread_info->transformed_data_v[i]),
                  data_type);
        MatToBlob(cv_label, *(cur_thread_info->transformed_label_v[i]),
                  label_type);
      }
      else if (this->transform_type_ == LayerParameter_TransformType_AUGMENT) {
        this->aug_data_transformer_->Transform(
            cv_data, cv_label, *(cur_thread_info->transformed_data_v[i]),
            *(cur_thread_info->transformed_label_v[i]));
      }
    }

    MergeBlob(cur_thread_info->transformed_data_v,
              cur_thread_info->data_);
    MergeBlob(cur_thread_info->transformed_label_v,
              cur_thread_info->label_);

    if (this->layer_param_.ex_image_data_param().debug_info()) {
      LOG(INFO) << "         Total: " << total_image_num;
      LOG(INFO) << " Encoded Cache: " << cached_data_map_encoded.size();
      LOG(INFO) << " Decoded Cache: " << cached_data_map_decoded.size();
    }

    cur_thread_info->end_.signal();
    if(this->layer_param_.ex_image_data_param().debug_info()) {
      LOG(INFO) << "thread ID " << index << " finish loading data";
    }
  }
}

template <typename Dtype>
void ExImageDataLayer<Dtype>::LoadData(const vector<string>& item,
                                       ExImageDataParameter_DataType data_type, int data_num, Mat& loaded_data) {
  if(data_type==ExImageDataParameter_DataType_IMAGE) {
    if(data_num==1) {
      ReadImageWithCache(item[0], new_height, new_width, new_max_size, is_color, loaded_data);
    }
    else if(is_color){
      vector<Mat> loaded_data_channels(data_num*3);
      for(int i=0;i<data_num;i++) {
        Mat loaded_data_temp;
        ReadImageWithCache(item[i], new_height, new_width, new_max_size, is_color, loaded_data_temp);
        split(loaded_data_temp, loaded_data_channels.data()+i*3);
      }
      merge(loaded_data_channels, loaded_data);
    }
    else {
      vector<Mat> loaded_data_channels(data_num);
      for(int i=0;i<data_num;i++) {
        ReadImageWithCache(item[i], new_height, new_width, new_max_size, is_color, loaded_data_channels[i]);
      }
      merge(loaded_data_channels, loaded_data);
    }
  }
  else {
    loaded_data.create(1, data_num, typeid(Dtype)==typeid(float)?CV_32F:CV_64F);
    memcpy(loaded_data.data, item[0].data(), data_num*sizeof(Dtype));
  }
}

template <typename Dtype>
void ExImageDataLayer<Dtype>::MatToBlob(const cv::Mat& mat, Blob<Dtype>& blob, ExImageDataParameter_DataType data_type) {
  if(data_type==ExImageDataParameter_DataType_IMAGE) {
    blob.Reshape(1, mat.channels(), mat.rows, mat.cols);
    vector<Mat> split_data(mat.channels());
    split(mat, &(split_data[0]));
    for(size_t j=0;j<split_data.size();j++) {
      split_data[j].convertTo(split_data[j], sizeof(Dtype)==4?CV_32F:CV_64F, 1, 0);
      caffe_copy(mat.cols*mat.rows, (Dtype*)split_data[j].data, blob.mutable_cpu_data()+mat.cols*mat.rows*j);
    }
  }
  else {
    vector<int> data_shape(2);
    data_shape[0]=1;
    data_shape[1]=mat.cols;
    blob.Reshape(data_shape);
    caffe_copy(mat.cols, (Dtype*)mat.data, blob.mutable_cpu_data());
  }
}

template <typename Dtype>
void ExImageDataLayer<Dtype>::MergeBlob(const std::vector<shared_ptr<Blob<Dtype>>>& blobs, Blob<Dtype>& merged_blob) {
CHECK_GT(blobs.size(), 0);
for(size_t i=0;i<blobs.size();i++) {
CHECK(blobs[i]->shape()==blobs[0]->shape())
<< "\n shape mismatch";
}
vector<int> blob_shape=blobs[0]->shape();
blob_shape[0]=blobs.size();
merged_blob.Reshape(blob_shape);
Dtype* blob_data = merged_blob.mutable_cpu_data();
for(size_t i=0;i<blobs.size();i++) {
caffe_copy(merged_blob.count(1), blobs[i]->cpu_data(), blob_data+merged_blob.offset(i));
}
}

// load images with cache, to accelerate
// if the new_height, new_width, is_color is same to the class's member,
// the read data will be cached.
// when access it again, no file I/O will be needed.
template <typename Dtype>
void ExImageDataLayer<Dtype>::ReadImageWithCache(const string& file_name,
                                                 int new_height, int new_width, int new_max_size, bool is_color, Mat& image) {
  bool use_cache=(new_height==this->new_height &&
      new_width==this->new_width &&
      new_max_size==this->new_max_size &&
      is_color==this->is_color &&
      cache_in_byte>0);
  if(use_cache) {
    if(cache_finished) {
      auto find_result_decoded=cached_data_map_decoded.find(file_name);
      if(find_result_decoded!=cached_data_map_decoded.end()) {
        image=(find_result_decoded->second).clone();
      }
      else{
        auto find_result_encoded=cached_data_map_encoded.find(file_name);
        if(find_result_encoded!=cached_data_map_encoded.end()) {
          const vector<unsigned char>& encoded_image=find_result_encoded->second;
          image=imdecode(encoded_image, -1);
        }
        else {
          ReadImageWithoutCache(file_name, new_height, new_width, new_max_size, is_color, image);
        }
      }
    }
    else {
      std::lock_guard<std::mutex> lock(mutex_cache_map);
      auto find_result_decoded=cached_data_map_decoded.find(file_name);
      if(find_result_decoded!=cached_data_map_decoded.end()) {
        image=(find_result_decoded->second).clone();
      }
      else{
        auto find_result_encoded=cached_data_map_encoded.find(file_name);
        if(find_result_encoded!=cached_data_map_encoded.end()) {
          const vector<unsigned char>& encoded_image=find_result_encoded->second;
          image=imdecode(encoded_image, -1);
          if(used_cache_in_byte<cache_in_byte &&
              cached_data_map_encoded.size()+cached_data_map_decoded.size()==total_image_num) {
            used_cache_in_byte+=image.cols*image.rows*image.channels()-encoded_image.size();
            cached_data_map_decoded.insert(make_pair(file_name, image.clone()));
            cached_data_map_encoded.erase(find_result_encoded);
          }
        }
        else {
          ReadImageWithoutCache(file_name, new_height, new_width, new_max_size, is_color, image);
          if(used_cache_in_byte < cache_in_byte) {
            string format=file_name.substr(file_name.find_last_of("."), string::npos);
            vector<unsigned char> encoded_img;
            CHECK(imencode(format, image, encoded_img));
            used_cache_in_byte+=encoded_img.size();
            cached_data_map_encoded.insert(make_pair(file_name, encoded_img));
          }
        }
      }
      if(cached_data_map_decoded.size()==total_image_num || used_cache_in_byte>=cache_in_byte) {
        cache_finished=true;
      }
    }
  }
  else {
    ReadImageWithoutCache(file_name, new_height, new_width, new_max_size, is_color, image);
  }
  CHECK(image.data!=NULL)
      << "\n load image from file failed: " << file_name;
}

// load images without cache
template <typename Dtype>
void ExImageDataLayer<Dtype>::ReadImageWithoutCache (const string& file_name,
                                                     int new_height, int new_width, int new_max_size, bool is_color, Mat& image) {
  try {
    image=imread(file_name);
  }
  catch(cv::Exception& e) {
    const char * s_ERROR=e.what();
    LOG(INFO) << s_ERROR << ": " << file_name;
    CHECK(false)
        << "\n load image from file failed: " << file_name;
  }
  if(is_color) {
    if(image.channels()==4){
      cvtColor(image, image, CV_BGRA2BGR);
    }
    else if(image.channels()==1) {
      cvtColor(image, image, CV_GRAY2BGR);
    }
  }
  else {
    if(image.channels()==4){
      cvtColor(image, image, CV_BGRA2GRAY);
    }
    else if(image.channels()==3) {
      cvtColor(image, image, CV_BGR2GRAY);
    }
  }
  if(new_height>0 || new_width>0) {
    Size new_size(new_width>0?new_width:image.cols, new_height>0?new_height:image.rows);
    resize(image, image, new_size);
  }
  else if(new_max_size>0) {
    if(image.cols>image.rows) {
      resize(image, image, Size(new_max_size, std::round((float)image.rows/image.cols*new_max_size)), 0, 0, INTER_NEAREST);
    }
    else {
      resize(image, image, Size(std::round((float)image.cols/image.rows*new_max_size), new_max_size), 0, 0, INTER_NEAREST);
    }
  }

  CHECK(image.data!=NULL)
      << "\n load image from file failed: " << file_name;
}

INSTANTIATE_CLASS(ExImageDataLayer);
REGISTER_LAYER_CLASS(ExImageData);

}  // namespace caffe
