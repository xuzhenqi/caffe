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

template<typename TypeParam>
class ShapeLossLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  ShapeLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_softmax_(new Blob<Dtype>()),
        blob_top_loss_(new Blob<Dtype>()) {
    vector<int> shape;
    shape.push_back(4);
    shape.push_back(3*2);
    blob_bottom_label_->Reshape(shape);
    shape[1] = 3;
    shape.push_back(5);
    shape.push_back(5);
    blob_bottom_data_->Reshape(shape);
    blob_top_softmax_->Reshape(shape);
    FillerParameter fillerParameter;
    fillerParameter.set_std(10);
    GaussianFiller<Dtype> filler(fillerParameter);
    filler.Fill(this->blob_bottom_data_);
    fillerParameter.set_min(0);
    fillerParameter.set_max(5);
    UniformFiller<Dtype> filler1(fillerParameter);
    filler1.Fill(blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
    blob_bottom_softmax_vec_.push_back(blob_bottom_data_);
    blob_top_softmax_vec_.push_back(blob_top_softmax_);
  }

  virtual ~ShapeLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
    delete blob_top_softmax_;
  }

  Blob<Dtype> *const blob_bottom_data_;
  Blob<Dtype> *const blob_bottom_label_;
  Blob<Dtype> *const blob_top_loss_;
  Blob<Dtype> *const blob_top_softmax_;
  vector<Blob<Dtype> *> blob_bottom_vec_;
  vector<Blob<Dtype> *> blob_top_vec_;
  vector<Blob<Dtype> *> blob_bottom_softmax_vec_;
  vector<Blob<Dtype> *> blob_top_softmax_vec_;
};

TYPED_TEST_CASE(ShapeLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(ShapeLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerParameter;
  layerParameter.mutable_softmax_param()->set_axis(2);
  ShapeLossLayer<Dtype> layer(layerParameter);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  SoftmaxLayer<Dtype> layer1(layerParameter);
  vector<int> shape;
  shape.push_back(4);
  shape.push_back(3);
  shape.push_back(25);
  this->blob_bottom_data_->Reshape(shape);
  layer1.SetUp(this->blob_bottom_softmax_vec_, this->blob_top_softmax_vec_);
  layer1.Forward(this->blob_bottom_softmax_vec_, this->blob_top_softmax_vec_);
  this->blob_top_softmax_->Reshape(4, 3, 5, 5);
  shape.clear();
  shape.push_back(4);
  shape.push_back(3);
  Blob<Dtype> mean_shape(shape); // data for row, diff for col
  Dtype *row_shape = mean_shape.mutable_cpu_data();
  Dtype *col_shape = mean_shape.mutable_cpu_diff();
  const Dtype *label = this->blob_bottom_label_->cpu_data();
  int channels = this->blob_bottom_data_->channels();
  Dtype loss = 0;
  for (int n = 0; n < this->blob_bottom_data_->num(); ++n) {
    for(int i = 0; i < channels; ++i) {
      row_shape[n*channels+i] = 0;
      col_shape[n*channels+i] = 0;
      for(int j = 0; j < 5; ++j) {
        for(int k = 0; k < 5; ++k) {
          row_shape[n*channels+i] +=
              j * this->blob_top_softmax_->data_at(n, i, j, k);
          col_shape[n*channels+i] +=
              k * this->blob_top_softmax_->data_at(n, i, j, k);
        }
      }
      loss += (label[2*(n*channels + i)] - col_shape[n*channels + i]) *
          (label[2*(n*channels + i)] - col_shape[n*channels + i]) +
          (label[2*(n*channels + i)+1] - row_shape[n*channels + i]) *
              (label[2*(n*channels+i)+1] - row_shape[n*channels+i]);
    }
  }
  CHECK_NEAR(this->blob_top_loss_->cpu_data()[0], loss / 4 / 3, 1e-5);
}

TYPED_TEST(ShapeLossLayerTest, TestForwardZero) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerParameter;
  layerParameter.mutable_softmax_param()->set_axis(2);
  SoftmaxLayer<Dtype> layer1(layerParameter);
  vector<int> shape;
  shape.push_back(4);
  shape.push_back(3);
  shape.push_back(25);
  this->blob_bottom_data_->Reshape(shape);
  layer1.SetUp(this->blob_bottom_softmax_vec_, this->blob_top_softmax_vec_);
  layer1.Forward(this->blob_bottom_softmax_vec_, this->blob_top_softmax_vec_);
  this->blob_top_softmax_->Reshape(4, 3, 5, 5);
  this->blob_bottom_data_->Reshape(4, 3, 5, 5);
  Dtype *label = this->blob_bottom_label_->mutable_cpu_data();
  int channels = this->blob_bottom_data_->channels();
  for (int n = 0; n < this->blob_bottom_data_->num(); ++n) {
    for(int i = 0; i < channels; ++i) {
      label[2*(n*channels+i)] = 0;
      label[2*(n*channels+i) + 1] = 0;
      for(int j = 0; j < 5; ++j) {
        for(int k = 0; k < 5; ++k) {
          label[2*(n*channels+i) + 1] +=
              j * this->blob_top_softmax_->data_at(n, i, j, k);
          label[2*(n*channels+i)] +=
              k * this->blob_top_softmax_->data_at(n, i, j, k);
        }
      }
    }
  }
  ShapeLossLayer<Dtype> layer(layerParameter);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  CHECK_NEAR(0, this->blob_top_loss_->cpu_data()[0], 1e-5);
}

TYPED_TEST(ShapeLossLayerTest, TestGradient){
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerParameter;
  layerParameter.mutable_softmax_param()->set_axis(2);
  ShapeLossLayer<Dtype> layer(layerParameter);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

} // namespace caffe
