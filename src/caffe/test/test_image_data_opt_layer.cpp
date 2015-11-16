#include <map>
#include <string>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ImageDataOptLayerTest : public MultiDeviceTest<TypeParam> {
 public:
  typedef typename TypeParam::Dtype Dtype;

 protected:
      ImageDataOptLayerTest(): seed_(1701) {}
  
  virtual ~ImageDataOptLayerTest() {
    delete blob_top_vec_[0];
    delete blob_top_vec_[1];
  }

  virtual void SetUp() {
    blob_top_vec_.push_back(new Blob<Dtype>());
    blob_top_vec_.push_back(new Blob<Dtype>());
    Caffe::set_random_seed(seed_);
    // TODO: make the test source available in test dir.
    source_ = string("/mnt/dataset3/small/source.txt");
    root_dir_ = string("/mnt/dataset3/small/optflow/");
  }
  
  int seed_;
  string source_;
  string root_dir_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ImageDataOptLayerTest, TestDtypesAndDevices);

TYPED_TEST(ImageDataOptLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_is_color(false);
  image_data_param->set_source(this->source_.c_str());
  image_data_param->set_shuffle(false);
  image_data_param->set_root_folder(this->root_dir_.c_str());
  ImageDataRNNParameter* image_data_rnn_param = param
      .mutable_image_data_rnn_param();
  image_data_rnn_param->set_fps(5);
  ImageDataOptLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->num(), 5);
  EXPECT_EQ(this->blob_top_vec_[0]->channels(), 10);
  EXPECT_EQ(this->blob_top_vec_[0]->height(), 256);
  EXPECT_EQ(this->blob_top_vec_[0]->width(), 256);
  EXPECT_EQ(this->blob_top_vec_[1]->count(), 5);
  for(int i=0; i<10; ++i)
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Generating Images
  DumpMatrixToTxt("temp/images.txt",
                  this->blob_top_vec_[0]->count(),
                  this->blob_top_vec_[0]->cpu_data(),
                  this->blob_top_vec_[0]->shape());
  DumpMatrixToTxt("temp/label.txt",
                this->blob_top_vec_[1]->count(),
                this->blob_top_vec_[1]->cpu_data(),
                this->blob_top_vec_[1]->shape());
}

}  // namespace caffe
