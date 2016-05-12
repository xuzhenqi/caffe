#include <cmath>
#include <vector>
#include <cfloat>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class KLDivergenceLossLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  KLDivergenceLossLayerTest()
      : pred_(new Blob<Dtype>(4, 3, 5, 5)),
        label_(new Blob<Dtype>(4, 3, 5, 5)),
        loss_(new Blob<Dtype>()){
    FillerParameter param;
    param.set_min(0.1);
    param.set_max(1.);
    UniformFiller<Dtype> filler(param);
    filler.Fill(pred_);
    filler.Fill(label_);
    Norm(pred_, 2);
    Norm(label_, 2);
    bottom_vec_.push_back(pred_);
    bottom_vec_.push_back(label_);
    top_vec_.push_back(loss_);
  }

  void Norm(Blob<Dtype> * blob, int axis) {
      Dtype* data = blob->mutable_cpu_data();
      const int dim = blob->count(axis);
      for(int i = 0; i < blob->count(0, axis); ++i) {
          Dtype sum = 0;
          for (int j = 0; j < dim; ++j) {
              sum += data[j];
          }
          for(int j = 0; j < dim; ++j) {
              data[j] /= sum;
          }
          data += dim;
      }
  }

  Blob<Dtype> *const pred_;
  Blob<Dtype> *const label_;
  Blob<Dtype> *const loss_;
  vector<Blob<Dtype>*> bottom_vec_;
  vector<Blob<Dtype>*> top_vec_;
};

TYPED_TEST_CASE(KLDivergenceLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(KLDivergenceLossLayerTest, TestReshape) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter param;
    param.mutable_kl_divergence_loss_param()->set_axis(2);
    KLDivergenceLossLayer<Dtype> layer(param);
    layer.SetUp(this->bottom_vec_, this->top_vec_);
    layer.Reshape(this->bottom_vec_, this->top_vec_);
    EXPECT_EQ(this->loss_->count(), 1);
}

TYPED_TEST(KLDivergenceLossLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter param;
    param.mutable_kl_divergence_loss_param()->set_axis(2);
    param.mutable_kl_divergence_loss_param()->set_margin(1e-5);
    KLDivergenceLossLayer<Dtype> layer(param);
    layer.SetUp(this->bottom_vec_, this->top_vec_);
    layer.Forward(this->bottom_vec_, this->top_vec_);
    Dtype loss = 0;
    for (int n = 0; n < this->label_->num(); ++n) {
        for (int c = 0; c < this->label_->channels(); ++c) {
            for (int h = 0; h < this->label_->height(); ++h) {
                for (int w = 0; w < this->label_->width(); ++w) {
                    loss += this->label_->data_at(n, c, h, w) *
                        (log(this->label_->data_at(n, c, h, w) + 1e-5)
                        - log(this->pred_->data_at(n, c, h, w) + 1e-5));
                }
            }
        }
    }
    loss /= this->label_->count(0, 2);
    EXPECT_NEAR(loss, this->loss_->cpu_data()[0], 1e-5);
}

TYPED_TEST(KLDivergenceLossLayerTest, TestForwardZero) {
    typedef typename TypeParam::Dtype Dtype;
    caffe_copy(this->pred_->count(), this->pred_->cpu_data(),
               this->label_->mutable_cpu_data());
    LayerParameter param;
    param.mutable_kl_divergence_loss_param()->set_axis(2);
    param.mutable_kl_divergence_loss_param()->set_margin(1e-5);
    KLDivergenceLossLayer<Dtype> layer(param);
    layer.SetUp(this->bottom_vec_, this->top_vec_);
    layer.Forward(this->bottom_vec_, this->top_vec_);
    EXPECT_NEAR(0, this->loss_->cpu_data()[0], 1e-5);
}

TYPED_TEST(KLDivergenceLossLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter param;
    param.mutable_kl_divergence_loss_param()->set_axis(2);
    param.mutable_kl_divergence_loss_param()->set_margin(1e-2);
    KLDivergenceLossLayer<Dtype> layer(param);
    GradientChecker<Dtype> checker(1e-5, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->bottom_vec_,
                                    this->top_vec_, 0);
}

} // namespace caffe