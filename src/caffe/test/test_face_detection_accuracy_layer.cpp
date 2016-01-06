#include <cfloat>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

inline float square(float x) {
  return x * x;
}

template <typename Dtype>
class FaceDetectionAccuracyLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  FaceDetectionAccuracyLayerTest()
    : blob_bottom_data_(new Blob<Dtype>(3, 68, 5, 6)),
      blob_bottom_label_(new Blob<Dtype>(3, 68, 5, 6)),
      blob_top_(new Blob<Dtype>()){
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~FaceDetectionAccuracyLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
  }

  void getMaxIndex(const Dtype *data, int height, int width, int &h, int &w) {
    Dtype max = data[0];
    h = 0;
    w = 0;
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        if (data[i*width + j] > max) {
          max = data[i*width + j];
          h = i;
          w = j;
        }
      }
    }
  }


  Blob<Dtype> * const blob_bottom_data_;
  Blob<Dtype> * const blob_bottom_label_;
  Blob<Dtype> * const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(FaceDetectionAccuracyLayerTest, TestDtypes);

TYPED_TEST(FaceDetectionAccuracyLayerTest, TestSetup) {
  LayerParameter layerParameter;
  FaceDetectionAccuracyLayer<TypeParam> layer(layerParameter);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(FaceDetectionAccuracyLayerTest, TestForwardZero) {
  FillerParameter fillerParameter;
  fillerParameter.set_std(1.);
  fillerParameter.set_mean(0.5);
  GaussianFiller<TypeParam> filler(fillerParameter);
  filler.Fill(this->blob_bottom_data_);
  caffe_copy(this->blob_bottom_data_->count(),
             this->blob_bottom_data_->cpu_data(),
             this->blob_bottom_label_->mutable_cpu_data());
  caffe_add_scalar(this->blob_bottom_data_->count(), TypeParam(0.5),
                   this->blob_bottom_label_->mutable_cpu_data());
  LayerParameter layerParameter;
  FaceDetectionAccuracyLayer<TypeParam> layer(layerParameter);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  CHECK_NEAR(this->blob_top_->cpu_data()[0], 0, 1e-6);
}

TYPED_TEST(FaceDetectionAccuracyLayerTest, TestForwardNonZero) {
  typedef TypeParam Dtype;
  FillerParameter fillerParameter;
  fillerParameter.set_std(1.);
  fillerParameter.set_mean(0.5);
  GaussianFiller<TypeParam> filler(fillerParameter);
  filler.Fill(this->blob_bottom_data_);
  filler.Fill(this->blob_bottom_label_);
  LayerParameter layerParameter;
  FaceDetectionAccuracyLayer<TypeParam> layer(layerParameter);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  TypeParam  err = 0, eye_dis;
  vector<int> heights_label(68), widths_label(68), heights_data(68),
      widths_data(68);
  const TypeParam * data = this->blob_bottom_data_->cpu_data();
  const TypeParam * label = this->blob_bottom_label_->cpu_data();
  for (int n = 0; n < this->blob_bottom_data_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_data_->channels(); ++c) {
      this->getMaxIndex(data + this->blob_bottom_data_->offset(n, c),
                        this->blob_bottom_data_->height(),
                        this->blob_bottom_data_->width(), heights_data[c],
                        widths_data[c]);
      this->getMaxIndex(label + this->blob_bottom_label_->offset(n, c),
                        this->blob_bottom_label_->height(),
                        this->blob_bottom_label_->width(), heights_label[c],
                        widths_label[c]);
    }
    Dtype left_eye_w = (widths_label[36] + widths_label[37] + widths_label[38] +
        widths_label[39] + widths_label[40] + widths_label[41]) / 6.;
    Dtype left_eye_h = (heights_label[36] + heights_label[37] +
        heights_label[38] + heights_label[39] + heights_label[40] +
        heights_label[41]) / 6.;
    Dtype right_eye_w = (widths_label[42] + widths_label[43] + widths_label[44]
        + widths_label[45] + widths_label[46] + widths_label[47]) / 6.;
    Dtype right_eye_h = (heights_label[42] + heights_label[43] +
        heights_label[44] + heights_label[45] + heights_label[46] +
        heights_label[47]) / 6.;
    eye_dis = sqrt(square(left_eye_h - right_eye_h) + square
        (left_eye_w - right_eye_w));
    for (int c = 0; c < this->blob_bottom_data_->channels(); ++c) {
      err += sqrt(square(heights_data[c]-heights_label[c])+square
          (widths_data[c]-widths_label[c])) / eye_dis;
    }
  }
  err = err / this->blob_bottom_data_->num() /
      this->blob_bottom_data_->channels() * 100;
  CHECK_NEAR(err, this->blob_top_->cpu_data()[0], 1e-5);
}


} // namespace caffe

