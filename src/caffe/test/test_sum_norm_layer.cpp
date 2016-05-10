#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SumNormLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SumNormLayerTest()
      : bottom_(new Blob<Dtype>(3, 6, 3, 4)),
        top_(new Blob<Dtype>()) {
    vector<int> shape;
    shape.push_back(3);
    shape.push_back(6);
    shape.push_back(12);
    this->bottom_->Reshape(shape);
    Caffe::set_random_seed(1701);
    FillerParameter fillerParameter;
    GaussianFiller<Dtype> filler(fillerParameter);
    Blob<Dtype> softmax_bottom(this->bottom_->shape());
    vector<Blob<Dtype> *> softmax_bottom_vec(1, &softmax_bottom);
    filler.Fill(&softmax_bottom);
    bottom_vec_.push_back(this->bottom_);
    top_vec_.push_back(this->top_);
    LayerParameter param;
    param.mutable_softmax_param()->set_axis(2);
    SoftmaxLayer<Dtype> softmaxLayer(param);
    softmaxLayer.SetUp(softmax_bottom_vec, this->bottom_vec_);
    softmaxLayer.Forward(softmax_bottom_vec, this->bottom_vec_);
    this->bottom_->Reshape(3, 6, 3, 4);
    for (int i = 0; i < this->bottom_->num(); ++i) {
      for (int j = 0; j < this->bottom_->channels(); ++j) {
        Dtype sum = 0;
        for( int h = 0; h < this->bottom_->height(); ++h) {
          for (int w = 0; w < this->bottom_->width(); ++w) {
            sum += this->bottom_->data_at(i, j, h, w);
          }
        }
        CHECK_NEAR(sum, Dtype(1), 1e-5);
      }
    }

  }

  ~SumNormLayerTest() {
    delete bottom_;
    delete top_;
  }

  Blob<Dtype>* const bottom_;
  Blob<Dtype>* const top_;
  vector<Blob<Dtype>*> bottom_vec_;
  vector<Blob<Dtype>*> top_vec_;
};

TYPED_TEST_CASE(SumNormLayerTest, TestDtypesAndDevices);

TYPED_TEST(SumNormLayerTest, TestParseRange) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerParameter;
  SumNormLayer<Dtype> sum_norm_layer(layerParameter);

  vector<int> range;
  string str("0");
  CHECK(sum_norm_layer.ParseRanges(str, range));
  CHECK_EQ(range.size(), 1);
  CHECK_EQ(range[0], 0);
  CHECK_EQ(sum_norm_layer.min_channels(), 1);

  str = "0-4";
  CHECK(sum_norm_layer.ParseRanges(str, range));
  CHECK_EQ(range.size(), 4);
  for (int i = 0; i < 4; ++i) {
    CHECK_EQ(range[i], i);
  }
  CHECK_EQ(sum_norm_layer.min_channels(), 4);

  str = "1-4,7,2-5,9";
  int ans[] = {1, 2, 3, 7, 2, 3, 4, 9}, sz = 8;
  CHECK(sum_norm_layer.ParseRanges(str,range));
  CHECK_EQ(range.size(), sz);
  for (int i = 0; i < sz; ++i) {
    CHECK_EQ(range[i], ans[i]);
  }
  CHECK_EQ(sum_norm_layer.min_channels(), 10);

  str = "1-3,5";
  CHECK(sum_norm_layer.ParseRanges(str, range));
  CHECK_EQ(range.size(), 3);
  CHECK_EQ(range[0], 1);
  CHECK_EQ(range[1], 2);
  CHECK_EQ(range[2], 5);
  CHECK_EQ(sum_norm_layer.min_channels(), 10);

  str = "12 ";
  CHECK(sum_norm_layer.ParseRanges(str, range) == false);
}

TYPED_TEST(SumNormLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  param.mutable_sum_norm_param()->add_channels("0,2-3,5,1-3"); // 5
  param.mutable_sum_norm_param()->add_channels("1,3-4,0"); // 3
  SumNormLayer<Dtype> layer(param);
  layer.SetUp(this->bottom_vec_, this->top_vec_);
  layer.Reshape(this->bottom_vec_, this->top_vec_);
  CHECK_EQ(this->top_->num(), 3);
  CHECK_EQ(this->top_->channels(), 2);
  CHECK_EQ(this->top_->height(), 3);
  CHECK_EQ(this->top_->width(), 4);
}

TYPED_TEST(SumNormLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  param.mutable_sum_norm_param()->add_channels("0,2-3,5,1-3");
  param.mutable_sum_norm_param()->add_channels("1,2-4,0");
  SumNormLayer<Dtype> layer(param);
  layer.SetUp(this->bottom_vec_, this->top_vec_);
  layer.Forward(this->bottom_vec_, this->top_vec_);

  int range0[] = {0, 2, 5, 1, 2};
  int c = 0;
  for (int n = 0; n < 3; ++n) {
    for (int h = 0; h < 3; ++h) {
      for (int w = 0; w < 4; ++w) {
        Dtype sum = 0;
        for (int bc = 0; bc < 5; ++bc) {
          Dtype val =  this->bottom_->cpu_data()[
              this->bottom_->offset(n, range0[bc], h, w)];
          sum += val;
        }
        CHECK_NEAR(sum / 5, this->top_->cpu_data()[
            this->top_->offset(n, c, h, w)], 1e-5);
      }
    }
  }
  int range1[] = {1,2,3,0};
  c = 1;
  for (int n = 0; n < 3; ++n) {
    for (int h = 0; h < 3; ++h) {
      for (int w = 0; w < 4; ++w) {
        Dtype sum = 0;
        for (int bc = 0; bc < 4; ++bc) {
          sum += this->bottom_->cpu_data()[
              this->bottom_->offset(n, range1[bc], h, w)];
        }
        CHECK_NEAR(sum / 4, this->top_->cpu_data()[
            this->top_->offset(n, c, h, w)], 1e-5);
      }
    }
  }
}

TYPED_TEST(SumNormLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  param.mutable_sum_norm_param()->add_channels("0,2-3,5,1-3");
  param.mutable_sum_norm_param()->add_channels("1,2-4,0");
  SumNormLayer<Dtype> layer(param);
  layer.SetUp(this->bottom_vec_, this->top_vec_);


  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->bottom_vec_, this->top_vec_);

}

}  // namespace caffe
