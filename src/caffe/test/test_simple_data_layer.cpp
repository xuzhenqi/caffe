#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class SimpleDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SimpleDataLayerTest()
    : seed_(1501),
      filename_("src/caffe/test/test_data/test_simple_data_layer.data"),
      top_data_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    top_vec_.push_back(top_data_);
    Caffe::set_random_seed(seed_);
  }
  virtual ~SimpleDataLayerTest() {
    delete top_data_;
  }
  int seed_;
  string filename_;
  Blob<Dtype>* const top_data_;
  vector<Blob<Dtype>*> bottom_vec_, top_vec_;
};

TYPED_TEST_CASE(SimpleDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(SimpleDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  int nums = 5;
  LayerParameter param;
  SimpleDataParameter* data_param = param.mutable_simple_data_param();
  data_param->set_batch_size(5);
  data_param->set_filename(this->filename_.c_str());
  data_param->set_shuffle(true);
  data_param->set_nums(nums);
  SimpleDataLayer<Dtype> layer(param);
  layer.SetUp(this->bottom_vec_, this->top_vec_);
  EXPECT_EQ(this->top_data_->num(), 5);
  EXPECT_EQ(this->top_data_->channels(), nums);
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->bottom_vec_, this->top_vec_);
    for (int i = 0; i < 5; ++i) {
      for(int j = 0; j < nums; ++j) {
        std::cout << this->top_data_->cpu_data()[i * nums + j] << " ";
      }
      std::cout << std::endl;
    }
  }
}

}  // namespace caffe
