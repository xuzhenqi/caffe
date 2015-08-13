/*
 * test_l2n_layer.cpp
 *
 * Created on: Oct 12, 2015
 *     Author: Xu Zhenqi from BUPT
 *     Email : xuzhenqi1993@gmail.com
 */
 
#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<cstring>
#include<vector>
 
#include"gtest/gtest.h"
 
#include"caffe/blob.hpp"
#include"caffe/common.hpp"
#include"caffe/filler.hpp"
#include"caffe/vision_layers.hpp"
 
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
 
namespace caffe {
 
template <typename TypeParam>
class L2NLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  L2NLayerTest() {
    blob_bottom_vec_.push_back(new Blob<Dtype>(5, 3, 5, 5));
    blob_top_vec_.push_back(new Blob<Dtype>(5, 3, 5, 5));
    FillerParameter filler_param;
    filler_param.set_min(-5.0);
    filler_param.set_max(5.0);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_vec_[0]);
  }
  ~L2NLayerTest() {
    delete blob_bottom_vec_[0];
    delete blob_top_vec_[0];
  }
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};
 
TYPED_TEST_CASE(L2NLayerTest,TestDtypesAndDevices);
 
TYPED_TEST(L2NLayerTest,TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  L2NLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_,this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_,this->blob_top_vec_);
  // manually compute to compare
  const Dtype *bottom = this->blob_bottom_vec_[0]->cpu_data(); 
  const Dtype *top = this->blob_top_vec_[0]->cpu_data();
  int num = 5;
  int dim = 75;
  Dtype sum;
  for (int i = 0; i < num; ++i) {
    sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += bottom[i * dim + j] * bottom[i * dim + j];
    }
    sum = sqrt(sum);
    for (int j = 0; j < dim; ++j) {
      EXPECT_NEAR(top[i * dim + j], bottom[i * dim + j] / sum, 1e-5);
    }
  }
}
 
TYPED_TEST(L2NLayerTest,TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  L2NLayer<Dtype>layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_,this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2,1e-2, 1701);
  checker.CheckGradientExhaustive(&layer,this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
 
}  // namespace caffe
