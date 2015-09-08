#include <map>
#include <string>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class TripletImageDataLayerTest : public MultiDeviceTest<TypeParam> {
 public:
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TripletImageDataLayerTest(): seed_(1701) {}
  
  virtual ~TripletImageDataLayerTest() {
    delete blob_top_vec_[0];
    delete blob_top_vec_[1];
    delete blob_top_vec_[2];
  }

  virtual void SetUp() {
    blob_top_vec_.push_back(new Blob<Dtype>());
    blob_top_vec_.push_back(new Blob<Dtype>());
    blob_top_vec_.push_back(new Blob<Dtype>());
    Caffe::set_random_seed(seed_);
    // TODO
    filename_ = string("/mnt/dataset2/CASIAWebFace/filelist_crop.txt");
    statics_ = string("/mnt/dataset2/CASIAWebFace/identities.txt");
    root_dir_ = string("/mnt/dataset2/CASIAWebFace/casia_crop/");

  }
  
  int seed_;
  string filename_;
  string statics_;
  string root_dir_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TripletImageDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(TripletImageDataLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(false);
  image_data_param->set_root_folder(this->root_dir_.c_str());
  TripletImageDataParameter *triplet_image_data_param =
      param.mutable_triplet_image_data_param();
  triplet_image_data_param->set_statics(this->statics_.c_str());
  TripletImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->num(), 5);
  EXPECT_EQ(this->blob_top_vec_[0]->channels(), 3);
  EXPECT_EQ(this->blob_top_vec_[0]->height(), 96);
  EXPECT_EQ(this->blob_top_vec_[0]->width(), 96);
  EXPECT_EQ(this->blob_top_vec_[1]->num(), 5);
  EXPECT_EQ(this->blob_top_vec_[1]->channels(), 3);
  EXPECT_EQ(this->blob_top_vec_[1]->height(), 96);
  EXPECT_EQ(this->blob_top_vec_[1]->width(), 96);
  EXPECT_EQ(this->blob_top_vec_[2]->num(), 5);
  EXPECT_EQ(this->blob_top_vec_[2]->channels(), 3);
  EXPECT_EQ(this->blob_top_vec_[2]->height(), 96);
  EXPECT_EQ(this->blob_top_vec_[2]->width(), 96);
  const vector<int>& label_size = layer.label_size();
  const vector<string>& lines = layer.lines();
  const vector<int>& labels = layer.labels();
  // label_size
  EXPECT_EQ(label_size.size(), 10576);
  EXPECT_EQ(label_size[0], 0);
  EXPECT_EQ(label_size[1], 15);
  EXPECT_EQ(label_size[10575], 494410);
  // lines
  EXPECT_EQ(lines.size(), 494410);
  // labels
  EXPECT_EQ(labels.size(), 494410);
  EXPECT_EQ(labels[14], 0);
  EXPECT_EQ(labels[494409], 10574);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Generating Images
  /*
  DumpMatrixToTxt("temp/anchor.txt",
                  this->blob_top_vec_[0]->count(),
                  this->blob_top_vec_[0]->cpu_data(),
                  this->blob_top_vec_[0]->shape());
  DumpMatrixToTxt("temp/positive.txt",
                  this->blob_top_vec_[1]->count(),
                  this->blob_top_vec_[1]->cpu_data(),
                  this->blob_top_vec_[1]->shape());
  DumpMatrixToTxt("temp/negative.txt",
                  this->blob_top_vec_[2]->count(),
                  this->blob_top_vec_[2]->cpu_data(),
                  this->blob_top_vec_[2]->shape());
  */
  // Test GetIndexs
  vector<vector<int> > indexs(4, vector<int>(3, 0));
  layer.GetIndexs(indexs);
  std::cout << "Generated indexs: " << std::endl;
  for(int i = 0; i < 4; ++i) {
    std::cout << indexs[i][0] << " " << indexs[i][1] << " " 
        << indexs[i][2] << std::endl;
    EXPECT_EQ(labels[indexs[i][0]], labels[indexs[i][1]]);
    EXPECT_NE(labels[indexs[i][0]], labels[indexs[i][2]]);
  }
}

}  // namespace caffe
