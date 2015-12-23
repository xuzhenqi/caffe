#include <algorithm>
#include <vector>
#include <fstream>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename TypeParam>
class FaceDetectionDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  FaceDetectionDataLayerTest() :
      points_(5),
      filename_(EXAMPLES_SOURCE_DIR "images/eye_loc_temp.txt"){
    Caffe::set_random_seed(1701);
    for (int i = 0; i < 5; ++i){
      blob_top_vec_.push_back(new Blob<Dtype>);
    }
    for (int i = 0; i < 2; ++i){
      blob_top_vec_no_scale_.push_back(new Blob<Dtype>);
    }
    std::ifstream source(filename_.c_str());
    string temp;
    int index = 0;
    while (source >> temp) {
      labels_.push_back(vector<int>(2 * points_, 0));
      for (int i = 0; i < 2 * points_; ++i) {
        source >> labels_[index][i];
      }
      ++index;
    }
  }

  virtual ~FaceDetectionDataLayerTest() {
    for (int i = 0; i < 5; ++i) {
      delete blob_top_vec_[i];
    }
    for (int i = 0; i < 2; ++i) {
      delete blob_top_vec_no_scale_[i];
    }
  }

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_no_scale_;
  vector<Blob<Dtype>*> blob_top_vec_;
  string filename_;
  vector<vector<int> > labels_;
  int points_;
};

TYPED_TEST_CASE(FaceDetectionDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(FaceDetectionDataLayerTest, TestForwardGaussMap) {
  typedef  typename TypeParam::Dtype Dtype;

  LayerParameter layerParameter;
  GaussMapParameter* gaussMapParameter = layerParameter
      .mutable_gaussmap_param();
  gaussMapParameter->set_std(4);
  gaussMapParameter->set_points(this->points_);
  ImageDataParameter* imageDataParameter = layerParameter
      .mutable_image_data_param();
  imageDataParameter->set_batch_size(2);
  imageDataParameter->set_source(this->filename_.c_str());
  imageDataParameter->set_root_folder(EXAMPLES_SOURCE_DIR "images/");
  imageDataParameter->set_shuffle(false);
  imageDataParameter->set_is_color(false);
  shared_ptr<FaceDetectionDataLayer<Dtype> > layer(
      new FaceDetectionDataLayer<Dtype>(layerParameter));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_no_scale_);
  EXPECT_EQ(this->blob_top_vec_no_scale_[0]->num(), 2);
  EXPECT_EQ(this->blob_top_vec_no_scale_[0]->channels(), 1);
  EXPECT_EQ(this->blob_top_vec_no_scale_[0]->height(), 128);
  EXPECT_EQ(this->blob_top_vec_no_scale_[0]->width(), 128);

  EXPECT_EQ(this->blob_top_vec_no_scale_[1]->num(), 2);
  EXPECT_EQ(this->blob_top_vec_no_scale_[1]->channels(), this->points_);
  EXPECT_EQ(this->blob_top_vec_no_scale_[1]->height(), 128);
  EXPECT_EQ(this->blob_top_vec_no_scale_[1]->width(), 128);

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_no_scale_);

  int loc_width, loc_height;
  Dtype peak;
  for (int n = 0; n < 2; ++n) {
    for (int c = 0; c < this->points_; ++c) {
      loc_width = 0;
      loc_height = 0;
      peak = 0;
      for (int h = 0; h < 128; ++h) {
        for (int w = 0; w < 128; ++w) {
          if (peak < this->blob_top_vec_no_scale_[1]->data_at(n, c, h, w)) {
            peak = this->blob_top_vec_no_scale_[1]->data_at(n, c, h, w);
            loc_width = w;
            loc_height = h;
          }
        }
      }
      std::cout << loc_width << " " << loc_height << endl;
      CHECK_EQ(this->labels_[n][2*c], loc_width);
      CHECK_EQ(this->labels_[n][2*c + 1], loc_height);
    }
  }

  Dtype sum;
  const Dtype *data;
  for (int n = 0; n< 2; ++n) {
    for (int c = 0; c < this->points_; ++c) {
      sum = 0;
      data = this->blob_top_vec_no_scale_[1]->cpu_data() +
          this->blob_top_vec_no_scale_[1]->offset(n, c);
      for (int i = 0; i < 128 * 128; ++i)
        sum += data[i];
      CHECK_NEAR(1, sum, 1e-5);
    }
  }
}

TYPED_TEST(FaceDetectionDataLayerTest, TestForwardScale) {
  typedef  typename TypeParam::Dtype Dtype;

  LayerParameter layerParameter;
  GaussMapParameter* gaussMapParameter = layerParameter
      .mutable_gaussmap_param();
  gaussMapParameter->set_std(4);
  gaussMapParameter->set_points(this->points_);
  gaussMapParameter->add_scale(4);
  gaussMapParameter->add_scale(8);
  gaussMapParameter->add_scale(16);
  ImageDataParameter* imageDataParameter = layerParameter
      .mutable_image_data_param();
  imageDataParameter->set_batch_size(2);
  imageDataParameter->set_source(this->filename_.c_str());
  imageDataParameter->set_root_folder(EXAMPLES_SOURCE_DIR "images/");
  imageDataParameter->set_shuffle(false);
  imageDataParameter->set_is_color(false);
  shared_ptr<FaceDetectionDataLayer<Dtype> > layer(
      new FaceDetectionDataLayer<Dtype>(layerParameter));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->num(), 2);
  EXPECT_EQ(this->blob_top_vec_[0]->channels(), 1);
  EXPECT_EQ(this->blob_top_vec_[0]->height(), 128);
  EXPECT_EQ(this->blob_top_vec_[0]->width(), 128);

  EXPECT_EQ(this->blob_top_vec_[1]->num(), 2);
  EXPECT_EQ(this->blob_top_vec_[1]->channels(), this->points_);
  EXPECT_EQ(this->blob_top_vec_[1]->height(), 128);
  EXPECT_EQ(this->blob_top_vec_[1]->width(), 128);

  EXPECT_EQ(this->blob_top_vec_[2]->num(), 2);
  EXPECT_EQ(this->blob_top_vec_[2]->channels(), this->points_);
  EXPECT_EQ(this->blob_top_vec_[2]->height(), 32);
  EXPECT_EQ(this->blob_top_vec_[2]->width(), 32);

  EXPECT_EQ(this->blob_top_vec_[3]->num(), 2);
  EXPECT_EQ(this->blob_top_vec_[3]->channels(), this->points_);
  EXPECT_EQ(this->blob_top_vec_[3]->height(), 16);
  EXPECT_EQ(this->blob_top_vec_[3]->width(), 16);

  EXPECT_EQ(this->blob_top_vec_[4]->num(), 2);
  EXPECT_EQ(this->blob_top_vec_[4]->channels(), this->points_);
  EXPECT_EQ(this->blob_top_vec_[4]->height(), 8);
  EXPECT_EQ(this->blob_top_vec_[4]->width(), 8);

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  int loc_width, loc_height;
  Dtype peak;
  for (int n = 0; n < 2; ++n) {
    for (int c = 0; c < this->points_; ++c) {
      loc_width = 0;
      loc_height = 0;
      peak = 0;
      for (int h = 0; h < 128; ++h) {
        for (int w = 0; w < 128; ++w) {
          if (peak < this->blob_top_vec_[1]->data_at(n, c, h, w)) {
            peak = this->blob_top_vec_[1]->data_at(n, c, h, w);
            loc_width = w;
            loc_height = h;
          }
        }
      }
      std::cout << loc_width << " " << loc_height << endl;
      CHECK_EQ(this->labels_[n][2*c], loc_width);
      CHECK_EQ(this->labels_[n][2*c + 1], loc_height);
    }
  }

  Dtype sum;
  const Dtype *data;
  for (int n = 0; n< 2; ++n) {
    for (int c = 0; c < this->points_; ++c) {
      sum = 0;
      data = this->blob_top_vec_[1]->cpu_data() +
          this->blob_top_vec_[1]->offset(n,c);
      for (int i = 0; i < 128*128; ++i)
        sum += data[i];
      CHECK_NEAR(1, sum, 1e-5);

      sum = 0;
      data = this->blob_top_vec_[2]->cpu_data() +
          this->blob_top_vec_[2]->offset(n,c);
      for (int i = 0; i < 32*32; ++i)
        sum += data[i];
      CHECK_NEAR(1, sum, 1e-5);

      sum = 0;
      data = this->blob_top_vec_[3]->cpu_data() +
          this->blob_top_vec_[3]->offset(n,c);
      for (int i = 0; i < 16*16; ++i)
        sum += data[i];
      CHECK_NEAR(1, sum, 1e-5);

      sum = 0;
      data = this->blob_top_vec_[4]->cpu_data() +
          this->blob_top_vec_[4]->offset(n,c);
      for (int i = 0; i < 8*8; ++i)
        sum += data[i];
      CHECK_NEAR(1, sum, 1e-5);
    }
  }

}

} // namespace caffe