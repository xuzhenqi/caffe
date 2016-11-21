#include <cfloat>
#include <vector>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/rng.hpp"
#include "gtest/gtest.h"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class AlignmentAccuracyLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  AlignmentAccuracyLayerTest()
      : data_(new Blob<Dtype>(689, 68, 2, 1)),
        label_(new Blob<Dtype>(689, 68, 2, 1)),
        comm_err_(new Blob<Dtype>()),
        challenge_err_(new Blob<Dtype>()),
        full_err_(new Blob<Dtype>()),
        label_file_("/mnt/dataset2/300W/autoencoder/test.txt"),
        data_file_("/home/xuzhenqi/research/gardenia/output/test_pred_v6_gt_1.2_err_mean_pca.txt") {
    load_blob(data_file_, data_);
    load_blob(label_file_, label_);
    bottom_.push_back(data_);
    bottom_.push_back(label_);
    top_.push_back(comm_err_);
    top_.push_back(challenge_err_);
    top_.push_back(full_err_);
  }
  void load_blob(const string filename, Blob<Dtype>* blob) {
    CHECK_EQ(blob->count(), 689 * 68 * 2);
    Dtype* data = blob->mutable_cpu_data();
    std::ifstream in(filename);
    std::string temp;
    for (int i = 0; i < 689; ++i) {
      in >> temp;
      for (int j = 0; j < 68 * 2; ++j)
        in >> *(data++);
    }
  }
  Blob<Dtype>* const data_;
  Blob<Dtype>* const label_;
  Blob<Dtype>* const comm_err_;
  Blob<Dtype>* const challenge_err_;
  Blob<Dtype>* const full_err_;
  vector<Blob<Dtype>*> bottom_, top_;
  string label_file_, data_file_;
};

TYPED_TEST_CASE(AlignmentAccuracyLayerTest, TestDtypes);

TYPED_TEST(AlignmentAccuracyLayerTest, TestOutcome) {
  LayerParameter param;
  AlignmentAccuracyLayer<TypeParam> layer(param);
  layer.SetUp(this->bottom_, this->top_);
  layer.Forward(this->bottom_, this->top_);
  std::cout << "comm_err: " << this->comm_err_->cpu_data()[0] << std::endl;
  std::cout << "challenge_err: " << this->challenge_err_->cpu_data()[0] << std::endl;
  std::cout << "full_err: " << this->full_err_->cpu_data()[0] << std::endl;
}

}  // namespace caffe
