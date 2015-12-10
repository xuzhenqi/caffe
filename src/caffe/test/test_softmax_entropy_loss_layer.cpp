#include <cmath>
#include <vector>
#include <cfloat>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class SoftmaxEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxEntropyLossLayerTest()
    : blob_bottom_data_(new Blob<Dtype>()),
      blob_bottom_label_(new Blob<Dtype>()),
      blob_top_softmax_(new Blob<Dtype>()),
      blob_top_loss_(new Blob<Dtype>()) {
    vector<int> shape;
    shape.push_back(4);
    shape.push_back(3);
    shape.push_back(25);
    blob_bottom_data_->Reshape(shape);
    blob_bottom_label_->Reshape(shape);
    blob_top_softmax_->Reshape(shape);
    FillerParameter fillerParameter;
    fillerParameter.set_std(10);
    GaussianFiller<Dtype> filler(fillerParameter);
    filler.Fill(this->blob_bottom_data_);
    fillerParameter.set_min(0);
    fillerParameter.set_max(1);
    UniformFiller<Dtype> filler1(fillerParameter);
    filler1.Fill(blob_bottom_label_);
    Dtype sum = 0;
    for (int n = 0; n < blob_bottom_label_->num(); ++n) {
      for (int c = 0; c < blob_bottom_label_->channels(); ++c) {
        sum = 0;
        for (int h = 0; h < blob_bottom_label_->height(); ++h) {
            sum += blob_bottom_label_->cpu_data()[
                blob_bottom_label_->offset(n, c, h)];
        }
        for (int h = 0; h < blob_bottom_label_->height(); ++h) {
            blob_bottom_label_->mutable_cpu_data()[
                blob_bottom_label_->offset(n, c, h)] /= sum;
        }
      }
    }
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
    blob_bottom_softmax_vec_.push_back(blob_bottom_data_);
    blob_top_softmax_vec_.push_back(blob_top_softmax_);
  }

  virtual ~SoftmaxEntropyLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  Blob<Dtype> *const blob_bottom_data_;
  Blob<Dtype> *const blob_bottom_label_;
  Blob<Dtype> *const blob_top_loss_;
  Blob<Dtype> *const blob_top_softmax_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_bottom_softmax_vec_;
  vector<Blob<Dtype>*> blob_top_softmax_vec_;
};

TYPED_TEST_CASE(SoftmaxEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxEntropyLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerParameter;
  layerParameter.mutable_softmax_param()->set_axis(2);
  SoftmaxEntropyLossLayer<Dtype> layer(layerParameter);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  SoftmaxLayer<Dtype> layer1(layerParameter);
  layer1.SetUp(this->blob_bottom_softmax_vec_, this->blob_top_softmax_vec_);
  layer1.Forward(this->blob_bottom_softmax_vec_, this->blob_top_softmax_vec_);
  Dtype loss = 0;
  for (int n = 0; n < this->blob_bottom_label_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_label_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_label_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_label_->width(); ++w) {
          loss -= this->blob_bottom_label_->data_at(n, c, h, w) * log(std::max
              (Dtype(FLT_MIN), this->blob_top_softmax_->data_at(n, c, h, w)));
        }
      }
    }
  }
  loss /= this->blob_bottom_label_->count(0, 2);
  CHECK_NEAR(loss, this->blob_top_loss_->cpu_data()[0], 1e-5);
}

TYPED_TEST(SoftmaxEntropyLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerParameter;
  //layerParameter.add_loss_weight(3);
  layerParameter.mutable_softmax_param()->set_axis(2);
  SoftmaxEntropyLossLayer<Dtype> layer(layerParameter);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

} // namespace caffe

