#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename Dtype>
void InnerProduct(const Blob<Dtype> &input, const Blob<Dtype> &weight,
                  Blob<Dtype> &output, const Blob<Dtype> *bias = NULL) {
  int M = input.num();
  int K = weight.channels();
  int N = weight.num();
  CHECK_EQ(M, output.num());
  CHECK_EQ(N, output.channels());
  if (bias)
    CHECK_EQ(N, bias->count());
  Dtype *output_data = output.mutable_cpu_data();
  const Dtype *input_data = input.cpu_data();
  const Dtype *weight_data = weight.cpu_data();
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      output_data[i*N + j] = 0;
      for (int k = 0; k < K; ++k) {
        output_data[i*N + j] += input_data[i*K + k] * weight_data[j*K + k];
      }
      if (bias)
        output_data[i*N + j] += bias->cpu_data()[j];
    }
  }
}

template <typename TypeParam>
class InnerProductRNNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  InnerProductRNNLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_nobatch_(new Blob<Dtype>(1, 2, 3, 4)),
        previous_(new Blob<Dtype>()),
        previous_out_(new Blob<Dtype>()),
        ref_blob_top_(new Blob<Dtype>()),
        end_mark_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~InnerProductRNNLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_nobatch_;
    delete blob_top_;
    delete previous_;
    delete previous_out_;
    delete end_mark_;
  }

  virtual Dtype GetObjAndGradient(const Layer<Dtype>& layer,
       const vector<Blob<Dtype>*>& top, int top_id, int top_data_id) {
    Dtype loss = 0;
    if (top_id < 0) {
      // the loss will be half of the sum of squares of all outputs
      for (int i = 0; i < top.size(); ++i) {
        Blob<Dtype>* top_blob = top[i];
        const Dtype* top_blob_data = top_blob->cpu_data();
        Dtype* top_blob_diff = top_blob->mutable_cpu_diff();
        int count = top_blob->count();
        for (int j = 0; j < count; ++j) {
          loss += top_blob_data[j] * top_blob_data[j];
        }
        // set the diff: simply the data.
        caffe_copy(top_blob->count(), top_blob_data, top_blob_diff);
      }
      loss /= 2.;
    } else {
      // the loss will be the top_data_id-th element in the top_id-th blob.
      for (int i = 0; i < top.size(); ++i) {
        Blob<Dtype>* top_blob = top[i];
        Dtype* top_blob_diff = top_blob->mutable_cpu_diff();
        caffe_set(top_blob->count(), Dtype(0), top_blob_diff);
      }
      const Dtype loss_weight = 2;
      loss = top[top_id]->cpu_data()[top_data_id] * loss_weight;
      top[top_id]->mutable_cpu_diff()[top_data_id] = loss_weight;
    }
    return loss;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const end_mark_;
  Blob<Dtype>* const blob_bottom_nobatch_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const previous_;
  Blob<Dtype>* const previous_out_;
  Blob<Dtype>* const ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(InnerProductRNNLayerTest, TestDtypesAndDevices);

TYPED_TEST(InnerProductRNNLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->end_mark_->Reshape(2, 1, 1, 1);
  this->blob_bottom_vec_.push_back(this->end_mark_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  shared_ptr<InnerProductRNNLayer<Dtype> > layer(
      new InnerProductRNNLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 10);
}

TYPED_TEST(InnerProductRNNLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->end_mark_->Reshape(2, 1, 1, 1);
  this->blob_bottom_vec_.push_back(this->end_mark_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_type("gaussian");
    shared_ptr<InnerProductRNNLayer<Dtype> > layer(
        new InnerProductRNNLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    this->previous_->ReshapeLike(*(this->blob_top_));
    this->previous_out_->ReshapeLike(*(this->blob_top_));
    this->ref_blob_top_->ReshapeLike(*(this->blob_top_));
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->previous_);
    Blob<Dtype> &layer_previous = layer->GetPrevious();
    caffe_copy(this->previous_->count(), this->previous_->cpu_data(), 
               layer_previous.mutable_cpu_data());
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    InnerProduct(*(this->blob_bottom_), *(layer->blobs()[0]), 
                 *(this->ref_blob_top_), layer->blobs()[2].get());
    InnerProduct(*(this->previous_), *(layer->blobs()[1]),
                 *(this->previous_out_));
    caffe_add(this->previous_->count(), this->previous_out_->cpu_data(),
              this->ref_blob_top_->cpu_data(), 
              this->ref_blob_top_->mutable_cpu_data());
    const Dtype* top_data = this->blob_top_->cpu_data();
    Dtype* ref_top_data = this->ref_blob_top_->mutable_cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      if (ref_top_data[i] < 0)
        ref_top_data[i] = 0;
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-5);
    }
    this->previous_->CopyFrom(*(this->blob_top_));
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    InnerProduct(*(this->blob_bottom_), *(layer->blobs()[0]), 
                 *(this->ref_blob_top_), layer->blobs()[2].get());
    InnerProduct(*(this->previous_), *(layer->blobs()[1]),
                 *(this->previous_out_));
    caffe_add(this->previous_->count(), this->previous_out_->cpu_data(),
              this->ref_blob_top_->cpu_data(), 
              this->ref_blob_top_->mutable_cpu_data());
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->mutable_cpu_data();
    for (int i = 0; i < count; ++i) {
      if (ref_top_data[i] < 0)
        ref_top_data[i] = 0;
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-5);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(InnerProductRNNLayerTest, TestForwardReset) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->end_mark_->Reshape(2, 1, 1, 1);
  (this->end_mark_->mutable_cpu_data())[1] = 1;
  this->blob_bottom_vec_.push_back(this->end_mark_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_type("gaussian");
    shared_ptr<InnerProductRNNLayer<Dtype> > layer(
        new InnerProductRNNLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    this->previous_->ReshapeLike(*(this->blob_top_));
    this->previous_out_->ReshapeLike(*(this->blob_top_));
    this->ref_blob_top_->ReshapeLike(*(this->blob_top_));
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->previous_);
    Blob<Dtype> &layer_previous = layer->GetPrevious();
    caffe_copy(this->previous_->count(), this->previous_->cpu_data(), 
               layer_previous.mutable_cpu_data());
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    InnerProduct(*(this->blob_bottom_), *(layer->blobs()[0]), 
                 *(this->ref_blob_top_), layer->blobs()[2].get());
    InnerProduct(*(this->previous_), *(layer->blobs()[1]),
                 *(this->previous_out_));
    caffe_add(this->previous_->count(), this->previous_out_->cpu_data(),
              this->ref_blob_top_->cpu_data(), 
              this->ref_blob_top_->mutable_cpu_data());
    const Dtype* top_data = this->blob_top_->cpu_data();
    Dtype* ref_top_data = this->ref_blob_top_->mutable_cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      if (ref_top_data[i] < 0)
        ref_top_data[i] = 0;
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-5);
    }
    this->previous_->CopyFrom(*(this->blob_top_));
    int dim = this->previous_->count() / this->previous_->num();
    caffe_set(dim, Dtype(0), this->previous_->mutable_cpu_data() 
              + this->previous_->offset(1));
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    InnerProduct(*(this->blob_bottom_), *(layer->blobs()[0]), 
                 *(this->ref_blob_top_), layer->blobs()[2].get());
    InnerProduct(*(this->previous_), *(layer->blobs()[1]),
                 *(this->previous_out_));
    caffe_add(this->previous_->count(), this->previous_out_->cpu_data(),
              this->ref_blob_top_->cpu_data(), 
              this->ref_blob_top_->mutable_cpu_data());
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->mutable_cpu_data();
    for (int i = 0; i < count; ++i) {
      if (ref_top_data[i] < 0)
        ref_top_data[i] = 0;
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-5);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}
TYPED_TEST(InnerProductRNNLayerTest, TestNoBatchForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_nobatch_);
  this->end_mark_->Reshape(1, 1, 1, 1);
  this->blob_bottom_vec_.push_back(this->end_mark_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_type("gaussian");
    shared_ptr<InnerProductRNNLayer<Dtype> > layer(
        new InnerProductRNNLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    this->previous_->ReshapeLike(*(this->blob_top_));
    this->previous_out_->ReshapeLike(*(this->blob_top_));
    this->ref_blob_top_->ReshapeLike(*(this->blob_top_));
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->previous_);
    Blob<Dtype> &layer_previous = layer->GetPrevious();
    layer_previous.CopyFrom(*(this->previous_));
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    InnerProduct(*(this->blob_bottom_nobatch_), *(layer->blobs()[0]), 
                 *(this->ref_blob_top_), layer->blobs()[2].get());
    InnerProduct(*(this->previous_), *(layer->blobs()[1]),
                 *(this->previous_out_));
    caffe_add(this->previous_->count(), this->previous_out_->cpu_data(),
              this->ref_blob_top_->cpu_data(), 
              this->ref_blob_top_->mutable_cpu_data());
    const Dtype* top_data = this->blob_top_->cpu_data();
    Dtype* ref_top_data = this->ref_blob_top_->mutable_cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      if (ref_top_data[i] < 0)
        ref_top_data[i] = 0;
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-5);
    }
    this->previous_->CopyFrom(*(this->blob_top_));
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    InnerProduct(*(this->blob_bottom_nobatch_), *(layer->blobs()[0]), 
                 *(this->ref_blob_top_), layer->blobs()[2].get());
    InnerProduct(*(this->previous_), *(layer->blobs()[1]),
                 *(this->previous_out_));
    caffe_add(this->previous_->count(), this->previous_out_->cpu_data(),
              this->ref_blob_top_->cpu_data(), 
              this->ref_blob_top_->mutable_cpu_data());
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->mutable_cpu_data();
    for (int i = 0; i < count; ++i) {
      if (ref_top_data[i] < 0)
        ref_top_data[i] = 0;
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-5);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(InnerProductRNNLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->end_mark_->Reshape(2, 1, 1, 1);
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->end_mark_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_type("gaussian");
    InnerProductRNNLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    this->previous_->ReshapeLike(*(this->blob_top_));
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->previous_);
    Blob<Dtype>& layer_previous = layer.GetPrevious();
    layer_previous.CopyFrom(*(this->previous_));
    vector<bool> propagate_down(1, true);
    Dtype stepsize_ = 0.001;
    Dtype threshold_ = 0.01;
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      vector<Blob<Dtype>*> blobs_to_check;
      for (int j = 0; j < layer.blobs().size(); ++j) {
        Blob<Dtype>* blob = layer.blobs()[j].get();
        caffe_set(blob->count(), Dtype(0), blob->mutable_cpu_diff());
        blobs_to_check.push_back(blob);
      }
      blobs_to_check.push_back(this->blob_bottom_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      this->GetObjAndGradient(layer, this->blob_top_vec_, 0, i);
      layer.Backward(this->blob_top_vec_, propagate_down, 
                     this->blob_bottom_vec_);
      layer_previous.CopyFrom(*(this->previous_));
      vector<shared_ptr<Blob<Dtype> > >
          computed_gradient_blobs(blobs_to_check.size());
      for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) {
        Blob<Dtype>* current_blob = blobs_to_check[blob_id];
        computed_gradient_blobs[blob_id].reset(new Blob<Dtype>());
        computed_gradient_blobs[blob_id]->ReshapeLike(*current_blob);
        const int count = blobs_to_check[blob_id]->count();
        const Dtype* diff = blobs_to_check[blob_id]->cpu_diff();
        Dtype* computed_gradients =
            computed_gradient_blobs[blob_id]->mutable_cpu_data();
        caffe_copy(count, diff, computed_gradients);
      }
      for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) {
        Blob<Dtype>* current_blob = blobs_to_check[blob_id];
        const Dtype* computed_gradients =
            computed_gradient_blobs[blob_id]->cpu_data();
        for (int feat_id = 0; feat_id < current_blob->count(); ++feat_id) {
          Dtype estimated_gradient = 0;
          Dtype positive_objective = 0;
          Dtype negative_objective = 0;
          current_blob->mutable_cpu_data()[feat_id] += stepsize_;
          layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
          layer_previous.CopyFrom(*(this->previous_));
          positive_objective =
              this->GetObjAndGradient(layer, this->blob_top_vec_, 0, i);
          // Compute loss with stepsize_ subtracted from input.
          current_blob->mutable_cpu_data()[feat_id] -= stepsize_ * 2;
          layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
          layer_previous.CopyFrom(*(this->previous_));
          negative_objective =
              this->GetObjAndGradient(layer, this->blob_top_vec_, 0, i);
          // Recover original input value.
          current_blob->mutable_cpu_data()[feat_id] += stepsize_;
          estimated_gradient = (positive_objective - negative_objective) /
              stepsize_ / 2.;
          Dtype computed_gradient = computed_gradients[feat_id];
          Dtype feature = current_blob->cpu_data()[feat_id];
          Dtype scale = std::max(
              std::max(fabs(computed_gradient), fabs(estimated_gradient)), 1.);
          EXPECT_NEAR(computed_gradient, estimated_gradient, threshold_ * scale)
              << "debug: (top_data_id, blob_id, feat_id)="
              << i << "," << blob_id << "," << feat_id
              << "; feat = " << feature
              << "; objective+ = " << positive_objective
              << "; objective- = " << negative_objective;

        }
      }
    }
    this->previous_->CopyFrom(*(this->blob_top_));
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      vector<Blob<Dtype>*> blobs_to_check;
      for (int j = 0; j < layer.blobs().size(); ++j) {
        Blob<Dtype>* blob = layer.blobs()[j].get();
        caffe_set(blob->count(), Dtype(0), blob->mutable_cpu_diff());
        blobs_to_check.push_back(blob);
      }
      blobs_to_check.push_back(this->blob_bottom_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      this->GetObjAndGradient(layer, this->blob_top_vec_, 0, i);
      layer.Backward(this->blob_top_vec_, propagate_down, 
                     this->blob_bottom_vec_);
      layer_previous.CopyFrom(*(this->previous_));
      vector<shared_ptr<Blob<Dtype> > >
          computed_gradient_blobs(blobs_to_check.size());
      for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) {
        Blob<Dtype>* current_blob = blobs_to_check[blob_id];
        computed_gradient_blobs[blob_id].reset(new Blob<Dtype>());
        computed_gradient_blobs[blob_id]->ReshapeLike(*current_blob);
        const int count = blobs_to_check[blob_id]->count();
        const Dtype* diff = blobs_to_check[blob_id]->cpu_diff();
        Dtype* computed_gradients =
            computed_gradient_blobs[blob_id]->mutable_cpu_data();
        caffe_copy(count, diff, computed_gradients);
      }
      for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) {
        Blob<Dtype>* current_blob = blobs_to_check[blob_id];
        const Dtype* computed_gradients =
            computed_gradient_blobs[blob_id]->cpu_data();
        for (int feat_id = 0; feat_id < current_blob->count(); ++feat_id) {
          Dtype estimated_gradient = 0;
          Dtype positive_objective = 0;
          Dtype negative_objective = 0;
          current_blob->mutable_cpu_data()[feat_id] += stepsize_;
          layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
          layer_previous.CopyFrom(*(this->previous_));
          positive_objective =
              this->GetObjAndGradient(layer, this->blob_top_vec_, 0, i);
          // Compute loss with stepsize_ subtracted from input.
          current_blob->mutable_cpu_data()[feat_id] -= stepsize_ * 2;
          layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
          layer_previous.CopyFrom(*(this->previous_));
          negative_objective =
              this->GetObjAndGradient(layer, this->blob_top_vec_, 0, i);
          // Recover original input value.
          current_blob->mutable_cpu_data()[feat_id] += stepsize_;
          estimated_gradient = (positive_objective - negative_objective) /
              stepsize_ / 2.;
          Dtype computed_gradient = computed_gradients[feat_id];
          Dtype feature = current_blob->cpu_data()[feat_id];
          Dtype scale = std::max(
              std::max(fabs(computed_gradient), fabs(estimated_gradient)), 1.);
          EXPECT_NEAR(computed_gradient, estimated_gradient, threshold_ * scale)
              << "debug: (top_data_id, blob_id, feat_id)="
              << i << "," << blob_id << "," << feat_id
              << "; feat = " << feature
              << "; objective+ = " << positive_objective
              << "; objective- = " << negative_objective;

        }
      }
    }

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
