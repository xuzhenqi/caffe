#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class GaussMapLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GaussMapLayerTest(): blob_bottom_(new Blob<Dtype>()),
                       blob_top_(new Blob<Dtype>()) {
    vector<int> shape;
    shape.push_back(5);
    blob_bottom_->Reshape(shape);
    Caffe::set_random_seed(1701);
    FillerParameter fillerParameter;
    fillerParameter.set_min(0);
    fillerParameter.set_max(119);
    UniformFiller<Dtype> filler(fillerParameter);
    filler.Fill(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~GaussMapLayerTest() { delete blob_bottom_; delete blob_top_; }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GaussMapLayerTest, TestDtypesAndDevices);

TYPED_TEST(GaussMapLayerTest, TestForward) {
  typedef  typename TypeParam::Dtype Dtype;
  LayerParameter layerParameter;
  GaussMapParameter* gaussMapParameter = layerParameter
      .mutable_gaussmap_param();
  gaussMapParameter->set_std(4);
  gaussMapParameter->set_width(10);
  gaussMapParameter->set_height(12);
  shared_ptr<GaussMapLayer<Dtype> > layer(new GaussMapLayer<Dtype>
                                              (layerParameter));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 5);
  EXPECT_EQ(this->blob_top_->channels(), 10);
  EXPECT_EQ(this->blob_top_->height(), 12);

  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  for (int i = 0; i < this->blob_top_->num(); ++i) {
    std::cout << int(bottom_data[i] + 0.5) << " ";
  }
  std::cout << std::endl;
  int width, height, peak;
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    const Dtype* top_data_n = top_data + this->blob_top_->offset(n);
    width = 0; height = 0; peak = *top_data_n;
    for (int w = 0; w < this->blob_top_->channels(); ++w) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        if (peak < top_data_n[w*this->blob_top_->height() + h]) {
          peak = top_data_n[w*this->blob_top_->height() + h];
          width = w;
          height = h;
        }
      }
    }
    CHECK_EQ(width, int(bottom_data[n] + 0.5) / this->blob_top_->height());
    CHECK_EQ(height, int(bottom_data[n] + 0.5) % this->blob_top_->height());
  }
}

} // namespace caffe