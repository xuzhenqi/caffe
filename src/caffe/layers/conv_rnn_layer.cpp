#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    this->kernel_h_ = this->kernel_w_ = conv_param.kernel_size();
  } else {
    this->kernel_h_ = conv_param.kernel_h();
    this->kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(this->kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(this->kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    this->pad_h_ = this->pad_w_ = conv_param.pad();
  } else {
    this->pad_h_ = conv_param.pad_h();
    this->pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    this->stride_h_ = this->stride_w_ = conv_param.stride();
  } else {
    this->stride_h_ = conv_param.stride_h();
    this->stride_w_ = conv_param.stride_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  this->is_1x1_ = this->kernel_w_ == 1 && this->kernel_h_ == 1
      && this->stride_h_ == 1 && this->stride_w_ == 1 && 
      this->pad_h_ == 0 && this->pad_w_ == 0;
  // Configure output channels and groups.
  this->channels_ = bottom[0]->channels();
  this->num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(this->num_output_, 0);
  this->group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(this->channels_ % this->group_, 0);
  CHECK_EQ(this->num_output_ % this->group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    //conv_out_channels_ = channels_;
    //conv_in_channels_ = num_output_;
    LOG(FATAL) << "Reverse_dimension is not supported";
  } else {
    this->conv_out_channels_ = this->num_output_;
    this->conv_in_channels_ = this->channels_;
    conv_in_channels_previous_ = this->num_output_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the filter weights between time steps
  // - blobs_[2] holds the biases (optional)
  // - blobs_[3] holds the biases between time steps (optional)
  this->bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(4);
    } else {
      this->blobs_.resize(2);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        this->conv_out_channels_, this->conv_in_channels_ / this->group_, 
        this->kernel_h_, this->kernel_w_));
    this->blobs_[1].reset(new Blob<Dtype>(
        this->conv_out_channels_, conv_in_channels_previous_ / this->group_, 
        this->kernel_h_, this->kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    weight_filler->Fill(this->blobs_[1].get());
    // If necessary, initialize and fill the biases.
    if (this->bias_term_) {
      vector<int> bias_shape(1, this->num_output_);
      this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
      this->blobs_[3].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[2].get());
      bias_filler->Fill(this->blobs_[3].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  neuron_layer_->LayerSetUp(top, top);
  Reshape(bottom, top);
  caffe_set(previous_.count(), Dtype(0), previous_.mutable_cpu_data());
}

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  this->num_ = bottom[0]->num();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();
  // Shape the tops.
  compute_output_shape();
  // Only do the same size convolution
  CHECK_EQ(this->height_, this->height_out_);
  CHECK_EQ(this->width_, this->width_out_);
  top[0]->Reshape(this->num_, this->num_output_, this->height_out_, 
                  this->width_out_);
  previous_out_.Reshape(this->num_, this->num_output_, this->height_out_, 
                        this->width_out_);
  previous_.Reshape(this->num_, this->num_output_, this->height_out_, 
                    this->width_out_);
  if (reverse_dimensions()) {
    //conv_in_height_ = height_out_;
    //conv_in_width_ = width_out_;
    //conv_out_spatial_dim_ = height_ * width_;
    LOG(FATAL) << "Reverse dimesion is not supported";
  } else {
    this->conv_in_height_ = this->height_;
    this->conv_in_width_ = this->width_;
    this->conv_out_spatial_dim_ = this->height_out_ * this->width_out_;
  }
  this->kernel_dim_ = this->conv_in_channels_ * this->kernel_h_ * 
      this->kernel_w_;
  kernel_dim_previous_ = conv_in_channels_previous_ * this->kernel_h_ * 
      this->kernel_w_;
  this->weight_offset_ = this->conv_out_channels_ * this->kernel_dim_ / 
      this->group_ / this->group_;
  weight_offset_previous_ = this->conv_out_channels_ * kernel_dim_previous_ / 
      this->group_ / this->group_;
  this->col_offset_ = this->kernel_dim_ * this->conv_out_spatial_dim_ / 
      this->group_;
  col_offset_previous_ = kernel_dim_previous_ * this->conv_out_spatial_dim_ / 
      this->group_;
  this->output_offset_ = this->conv_out_channels_ * 
      this->conv_out_spatial_dim_ / this->group_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  if (reverse_dimensions()) {
    LOG(FATAL) << "Reverse dimesion is not supported";
    //col_buffer_.Reshape(1, kernel_dim_, height_, width_);
  } else {
    this->col_buffer_.Reshape(1, this->kernel_dim_, this->height_out_, 
                              this->width_out_);
    col_buffer_previous_.Reshape(1, kernel_dim_previous_, this->height_out_, 
                                 this->width_out_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (this->bias_term_) {
    vector<int> bias_multiplier_shape(1, this->height_out_ * this->width_out_);
    this->bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(this->bias_multiplier_.count(), Dtype(1),
        this->bias_multiplier_.mutable_cpu_data());
  }
  neuron_layer_->Reshape(top, top);
}

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::forward_rnn_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_rnn_cpu(input, col_buffer_previous_.mutable_cpu_data());
    }
    col_buff = col_buffer_previous_.cpu_data();
  }
  for (int g = 0; g < this->group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->conv_out_channels_ /
        this->group_, this->conv_out_spatial_dim_, 
        kernel_dim_previous_ / this->group_,
        (Dtype)1., weights + weight_offset_previous_ * g, 
        col_buff + col_offset_previous_ * g,
        (Dtype)0., output + this->output_offset_ * g);
  }
}

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::backward_rnn_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_previous_.mutable_cpu_data();
  if (this->is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < this->group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 
        kernel_dim_previous_ / this->group_,
        this->conv_out_spatial_dim_, this->conv_out_channels_ / this->group_,
        (Dtype)1., weights + weight_offset_previous_ * g, 
        output + this->output_offset_ * g,
        (Dtype)0., col_buff + col_offset_previous_ * g);
  }
  if (!this->is_1x1_) {
    conv_col2im_rnn_cpu(col_buff, input);
  }
}

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::weight_rnn_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!this->is_1x1_) {
    conv_im2col_rnn_cpu(input, col_buffer_previous_.mutable_cpu_data());
    col_buff = col_buffer_previous_.cpu_data();
  }
  for (int g = 0; g < this->group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 
        this->conv_out_channels_ / this->group_,
        kernel_dim_previous_ / this->group_, this->conv_out_spatial_dim_,
        (Dtype)1., output + this->output_offset_ * g, 
        col_buff + col_offset_previous_ * g,
        (Dtype)1., weights + weight_offset_previous_ * g);
  }
}

#ifndef CPU_ONLY

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::forward_rnn_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_rnn_gpu(input, col_buffer_previous_.mutable_gpu_data());
    }
    col_buff = col_buffer_previous_.gpu_data();
  }
  for (int g = 0; g < this->group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 
        this->conv_out_channels_ / this->group_, this->conv_out_spatial_dim_, 
        kernel_dim_previous_ / this->group_,
        (Dtype)1., weights + weight_offset_previous_ * g, 
        col_buff + col_offset_previous_ * g,
        (Dtype)0., output + this->output_offset_ * g);
  }
}

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::backward_rnn_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_previous_.mutable_gpu_data();
  if (this->is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < this->group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 
        kernel_dim_previous_ / this->group_,
        this->conv_out_spatial_dim_, this->conv_out_channels_ / this->group_,
        (Dtype)1., weights + weight_offset_previous_ * g, 
        output + this->output_offset_ * g,
        (Dtype)0., col_buff + col_offset_previous_ * g);
  }
  if (!this->is_1x1_) {
    conv_col2im_rnn_gpu(col_buff, input);
  }
}

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::weight_rnn_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!this->is_1x1_) {
    conv_im2col_rnn_gpu(input, col_buffer_previous_.mutable_gpu_data());
    col_buff = col_buffer_previous_.gpu_data();
  }
  for (int g = 0; g < this->group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 
        this->conv_out_channels_ / this->group_,
        kernel_dim_previous_ / this->group_, this->conv_out_spatial_dim_,
        (Dtype)1., output + this->output_offset_ * g, 
        col_buff + col_offset_previous_ * g,
        (Dtype)1., weights + weight_offset_previous_ * g);
  }
}

#endif // !CPU_ONLY

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  //DumpMatrixToTxt("temp/weight_0", *(this->blobs_[0]));
  //DumpMatrixToTxt("temp/bottom", *(bottom[0]));
  for (int n = 0; n < this->num_; ++n) {
    this->forward_cpu_gemm(bottom_data + bottom[0]->offset(n), weight,
        top_data + top[0]->offset(n));
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[2]->cpu_data();
      this->forward_cpu_bias(top_data + top[0]->offset(n), bias);
    }
  }
  //DumpMatrixToTxt("temp/top_1", *(top[0]));
  //DumpMatrixToTxt("temp/bias_0", *(this->blobs_[2]));
  //DumpMatrixToTxt("temp/bias_1", *(this->blobs_[3]));
  //DumpMatrixToTxt("temp/weight_1", *(this->blobs_[1]));
  //DumpMatrixToTxt("temp/previous", previous_);
  weight = this->blobs_[1]->cpu_data();
  top_data = previous_out_.mutable_cpu_data();
  bottom_data = previous_.cpu_data();
  for (int n = 0; n < this->num_; ++n) {
    this->forward_rnn_cpu_gemm(bottom_data + previous_.offset(n), weight,
        top_data + previous_out_.offset(n));
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[3]->cpu_data();
      this->forward_cpu_bias(top_data + previous_out_.offset(n), bias);
    }
  }
  //DumpMatrixToTxt("temp/previous_out", previous_out_);
  //caffe_copy(previous_.count(), previous_out_.cpu_data(), 
  //           previous_out_.mutable_cpu_diff());
  caffe_add(previous_.count(), previous_out_.cpu_data(), top[0]->cpu_data(),
            top[0]->mutable_cpu_data());
  neuron_layer_->Forward(top, top);
  //DumpMatrixToTxt("temp/top_2", *(top[0]));
  caffe_copy(previous_.count(), previous_.cpu_data(), 
             previous_out_.mutable_cpu_data());
  caffe_copy(previous_.count(), top[0]->cpu_data(), 
             previous_.mutable_cpu_data());

}

template <typename Dtype>
void ConvolutionRNNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  vector<bool> temp(top.size(), true);
  neuron_layer_->Backward(top, temp, top);
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[2]) {
    Dtype* bias_diff = this->blobs_[2]->mutable_cpu_diff();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_cpu_bias(bias_diff, top_diff + top[0]->offset(n));
    }
  }
  if (this->param_propagate_down_[0] || propagate_down[0]) {
    for (int n = 0; n < this->num_; ++n) {
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0]) {
        this->weight_cpu_gemm(bottom_data + bottom[0]->offset(n),
            top_diff + top[0]->offset(n), weight_diff);
      }
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[0]) {
        this->backward_cpu_gemm(top_diff + top[0]->offset(n), weight,
            bottom_diff + bottom[0]->offset(n));
      }
    }
  }
  
  weight = this->blobs_[1]->cpu_data();
  weight_diff = this->blobs_[1]->mutable_cpu_diff();
  top_diff = top[0]->cpu_diff();
  bottom_data = previous_out_.cpu_data();
  // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[3]) {
    Dtype* bias_diff = this->blobs_[3]->mutable_cpu_diff();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_cpu_bias(bias_diff, top_diff + top[0]->offset(n));
    }
  }
  if (this->param_propagate_down_[1] || propagate_down[0]) {
    for (int n = 0; n < this->num_; ++n) {
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[1]) {
        this->weight_rnn_cpu_gemm(bottom_data + previous_out_.offset(n),
            top_diff + top[0]->offset(n), weight_diff);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionRNNLayer);
#endif

INSTANTIATE_CLASS(ConvolutionRNNLayer);

}  // namespace caffe
