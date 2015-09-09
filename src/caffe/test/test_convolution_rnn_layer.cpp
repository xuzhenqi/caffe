#include <cstring>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
                const shared_ptr<Blob<Dtype> >& weights, 
                const shared_ptr<Blob<Dtype> >&bias,
                Blob<Dtype>* out) {
  // Kernel size, stride, and pad
  caffe_set(out->count(), Dtype(0), out->mutable_cpu_data());
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_size()) {
    kernel_h = kernel_w = conv_param->kernel_size();
  } else {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  }
  int pad_h, pad_w;
  if (!conv_param->has_pad_h()) {
    pad_h = pad_w = conv_param->pad();
  } else {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  }
  int stride_h, stride_w;
  if (!conv_param->has_stride_h()) {
    stride_h = stride_w = conv_param->stride();
  } else {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->channels() / groups;
  int k_g = in->channels() / groups;
  int o_head, k_head;
  // Convolution
  const Dtype* in_data = in->cpu_data();
  const Dtype* weight_data = weights->cpu_data();
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->num(); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int y = 0; y < out->height(); y++) {
            for (int x = 0; x < out->width(); x++) {
              for (int p = 0; p < kernel_h; p++) {
                for (int q = 0; q < kernel_w; q++) {
                  int in_y = y * stride_h - pad_h + p;
                  int in_x = x * stride_w - pad_w + q;
                  if (in_y >= 0 && in_y < in->height()
                      && in_x >= 0 && in_x < in->width()) {
                    out_data[out->offset(n, o + o_head, y, x)] +=
                        in_data[in->offset(n, k + k_head, in_y, in_x)]
                        * weight_data[weights->offset(o + o_head, k, p, q)];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = bias->cpu_data();
    for (int n = 0; n < out->num(); n++) {
      for (int o = 0; o < out->channels(); o++) {
        for (int y = 0; y < out->height(); y++) {
          for (int x = 0; x < out->width(); x++) {
            out_data[out->offset(n, o, y, x)] += bias_data[o];
          }
        }
      }
    }
  }
}

template void caffe_conv(const Blob<float>* in,
ConvolutionParameter* conv_param,
const shared_ptr<Blob<float> > & weights,
const shared_ptr<Blob<float> > & bias,
Blob<float>* out);
template void caffe_conv(const Blob<double>* in,
ConvolutionParameter* conv_param,
const shared_ptr<Blob<double> > & weights,
const shared_ptr<Blob<double> > & bias,
Blob<double>* out);

template <typename TypeParam>
class ConvolutionRNNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConvolutionRNNLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
      end_mark_(new Blob<Dtype>(2, 1, 1, 1)),
      previous_(new Blob<Dtype>()),
      previous_out_(new Blob<Dtype>()),
      blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(end_mark_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ConvolutionRNNLayerTest() {
    delete blob_bottom_;
    delete end_mark_;
    delete blob_top_;
    delete previous_;
    delete previous_out_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
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
  Blob<Dtype>* const previous_;
  Blob<Dtype>* const previous_out_;
  Blob<Dtype>* const blob_top_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConvolutionRNNLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConvolutionRNNLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(1);
  convolution_param->set_pad(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionRNNLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 4);
  std::cout << "test set up" << std::endl;
}

TYPED_TEST(ConvolutionRNNLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(1);
  convolution_param->set_pad(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionRNNLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  /*
  for (int i=0; i<4; ++i){
    std::cout << "blobs " << i << ": " << layer->blobs()[i]->num() << " "
        << layer->blobs()[i]->channels() << " "
        << layer->blobs()[i]->height() << " "
        << layer->blobs()[i]->width() << std::endl;
  }
  std::cout << "layer forward" << std::endl;
  */
  // Check against reference convolution.
  const Dtype* top_data;
  Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs()[0], 
             layer->blobs()[2],
             this->MakeReferenceTop(this->blob_top_));
  this->previous_->ReshapeLike(*(this->blob_top_));
  caffe_set(this->previous_->count(), Dtype(0), 
            this->previous_->mutable_cpu_data());
  this->previous_out_->ReshapeLike(*(this->blob_top_));
  caffe_conv(this->previous_, convolution_param, layer->blobs()[1], 
             layer->blobs()[3],
             this->previous_out_);
  caffe_add(this->previous_->count(), this->previous_out_->cpu_data(),
            this->ref_blob_top_->cpu_data(),
            this->ref_blob_top_->mutable_cpu_data());
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->mutable_cpu_data();

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    if (ref_top_data[i] < 0)
      ref_top_data[i] = 0;
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    EXPECT_NEAR(ref_top_data[i], ((ConvolutionRNNLayer<Dtype>*)layer.get())
                ->get_previous().cpu_data()[i], 1e-4);
  }

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_copy(this->previous_->count(), this->ref_blob_top_->cpu_data(), 
             this->previous_->mutable_cpu_data());
  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs()[0], 
             layer->blobs()[2],
             this->MakeReferenceTop(this->blob_top_));
  caffe_conv(this->previous_, convolution_param, layer->blobs()[1], 
             layer->blobs()[3],
             this->previous_out_);
  caffe_add(this->previous_->count(), this->previous_out_->cpu_data(),
            this->ref_blob_top_->cpu_data(),
            this->ref_blob_top_->mutable_cpu_data());
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->mutable_cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    if (ref_top_data[i] < 0)
      ref_top_data[i] = 0;
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvolutionRNNLayerTest, TestSimpleConvolutionReset) {
  typedef typename TypeParam::Dtype Dtype;
  this->end_mark_->mutable_cpu_data()[1] = 1;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(1);
  convolution_param->set_pad(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionRNNLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  /*
  for (int i=0; i<4; ++i){
    std::cout << "blobs " << i << ": " << layer->blobs()[i]->num() << " "
        << layer->blobs()[i]->channels() << " "
        << layer->blobs()[i]->height() << " "
        << layer->blobs()[i]->width() << std::endl;
  }
  std::cout << "layer forward" << std::endl;
  */
  // Check against reference convolution.
  const Dtype* top_data;
  Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs()[0], 
             layer->blobs()[2],
             this->MakeReferenceTop(this->blob_top_));
  this->previous_->ReshapeLike(*(this->blob_top_));
  caffe_set(this->previous_->count(), Dtype(0), 
            this->previous_->mutable_cpu_data());
  this->previous_out_->ReshapeLike(*(this->blob_top_));
  caffe_conv(this->previous_, convolution_param, layer->blobs()[1], 
             layer->blobs()[3],
             this->previous_out_);
  caffe_add(this->previous_->count(), this->previous_out_->cpu_data(),
            this->ref_blob_top_->cpu_data(),
            this->ref_blob_top_->mutable_cpu_data());
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->mutable_cpu_data();

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    if (ref_top_data[i] < 0)
      ref_top_data[i] = 0;
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }

  caffe_copy(this->previous_->count(), this->ref_blob_top_->cpu_data(), 
             this->previous_->mutable_cpu_data());
  int dim = this->previous_->count() / this->previous_->num();
  caffe_set(dim, Dtype(0), this->previous_->mutable_cpu_data() 
            + this->previous_->offset(1));
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs()[0], 
             layer->blobs()[2],
             this->MakeReferenceTop(this->blob_top_));
  caffe_conv(this->previous_, convolution_param, layer->blobs()[1], 
             layer->blobs()[3],
             this->previous_out_);
  caffe_add(this->previous_->count(), this->previous_out_->cpu_data(),
            this->ref_blob_top_->cpu_data(),
            this->ref_blob_top_->mutable_cpu_data());
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->mutable_cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    if (ref_top_data[i] < 0)
      ref_top_data[i] = 0;
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvolutionRNNLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(1);
  convolution_param->set_pad(1);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  ConvolutionRNNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  this->previous_->ReshapeLike(*(this->blob_top_));
  FillerParameter filler_param;
  filler_param.set_value(1.);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->previous_);
  Blob<Dtype>& layer_previous = layer.get_previous();
  layer_previous.CopyFrom(*(this->previous_));
  vector<bool> propagate_down(1, true);
  Dtype stepsize_ = 0.001;
  Dtype threshold_ = 0.01;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    vector<Blob<Dtype>*> blobs_to_check;
    for (int i = 0; i < layer.blobs().size(); ++i) {
      Blob<Dtype>* blob = layer.blobs()[i].get();
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
}

} // namespace caffe
