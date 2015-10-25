#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SumRNNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SumRNNLayerTest()
      : bottom_(new Blob<Dtype>(5, 3, 4, 5)),
        bottom_rnn_(new Blob<Dtype>(5, 3, 4, 5)),
        begin_marker_(new Blob<Dtype>(5, 1, 1, 1)),
        top_(new Blob<Dtype>(5, 3, 4, 5)) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_mean(0.5);
    filler_param.set_std(1);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->bottom_);
    filler.Fill(this->bottom_rnn_);
    filler.Fill(this->begin_marker_);
    bottom_vec_.push_back(bottom_);
    bottom_vec_.push_back(bottom_rnn_);
    bottom_vec_.push_back(begin_marker_);
    top_vec_.push_back(top_);
  }
  Blob<Dtype>* const bottom_;
  Blob<Dtype>* const bottom_rnn_;
  Blob<Dtype>* const begin_marker_;
  Blob<Dtype>* const top_;
  vector<Blob<Dtype>*> bottom_vec_;
  vector<Blob<Dtype>*> top_vec_;
};

TYPED_TEST_CASE(SumRNNLayerTest, TestDtypesAndDevices);

TYPED_TEST(SumRNNLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<SumRNNLayer<Dtype> > layer(new SumRNNLayer<Dtype>(layer_param));
  layer->SetUp(this->bottom_vec_, this->top_vec_);
  const vector<Dtype> coeffs = layer->GetCoeffs();
  EXPECT_EQ(this->top_->num(), 5);
  EXPECT_EQ(this->top_->channels(), 3);
  EXPECT_EQ(this->top_->height(), 4);
  EXPECT_EQ(this->top_->width(), 5);
  EXPECT_EQ(coeffs.size(), 2);
  EXPECT_NEAR(coeffs[0], 0.5, 1e-5);
  EXPECT_NEAR(coeffs[1], 0.5, 1e-5);
}

TYPED_TEST(SumRNNLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  shared_ptr<SumRNNLayer<Dtype> > layer(new SumRNNLayer<Dtype>(layer_param));
  layer->SetUp(this->bottom_vec_, this->top_vec_);
  const vector<Dtype> coeffs = layer->GetCoeffs();
  layer->Forward(this->bottom_vec_, this->top_vec_);
  const Dtype* data = this->top_->cpu_data();
  const int count = this->top_->count();
  const int dim = count / this->begin_marker_->count();
  const Dtype* bottom_data = this->bottom_->cpu_data();
  const Dtype* bottom_data_rnn = this->bottom_rnn_->cpu_data();
  const Dtype* begin_marker = this->begin_marker_->cpu_data();
  for (int i = 0; i < count; ++i) {
    if (begin_marker[i / dim] <= 0.5) {
      EXPECT_NEAR(data[i], 0.5 * bottom_data[i] + 0.5 * bottom_data_rnn[i],
                  1e-5);
    } else {
      EXPECT_NEAR(data[i], bottom_data[i], 1e-5);
    }
  }
}

TYPED_TEST(SumRNNLayerTest, TestGradients) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SumRNNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->bottom_vec_, this->top_vec_, 0);
  checker.CheckGradientEltwise(&layer, this->bottom_vec_, this->top_vec_, 1);
}

} // namespace caffe
