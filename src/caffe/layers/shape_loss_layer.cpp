#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ShapeLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  vector<int> shape;
  shape.push_back(bottom[0]->num());
  shape.push_back(bottom[0]->channels() * 2);
  mean_shape_.Reshape(shape);
  shape[0] = height_;
  shape[1] = width_;
  multiplier_.Reshape(shape);
  Dtype *row = multiplier_.mutable_cpu_data();
  Dtype *col = multiplier_.mutable_cpu_diff();
  for (int i = 0; i < multiplier_.shape(0); ++i) {
    for (int j = 0; j < multiplier_.shape(1); ++j) {
      *(col++) = j;
      *(row++) = i;
    }
  }
}

template <typename Dtype>
void ShapeLossLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels()*2, bottom[1]->channels());
  vector<int> shape;
  shape.push_back(bottom[0]->num());
  shape.push_back(bottom[0]->channels());
  shape.push_back(height_ * width_);
  bottom[0]->Reshape(shape);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void ShapeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* row = multiplier_.cpu_data();
  const Dtype* col = multiplier_.cpu_diff();
  Dtype* mean_row = mean_shape_.mutable_cpu_data();
  Dtype* mean_col = mean_shape_.mutable_cpu_diff();
  int dim = prob_.count() / outer_num_;
  caffe_cpu_gemv(CblasNoTrans, outer_num_, dim, Dtype(1.), prob_data, row,
                 Dtype(0.), mean_row);
  caffe_cpu_gemv(CblasNoTrans, outer_num_, dim, Dtype(1.), prob_data, col,
                 Dtype(0.), mean_col);
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    loss += square(*(label++) - *(mean_col++)) + square(*(label++) -
        *(mean_row++));
  }
  top[0]->mutable_cpu_data()[0] = loss / outer_num_;
}

template <typename Dtype>
void ShapeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                         const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* mean_row = mean_shape_.cpu_data();
  const Dtype* mean_col = mean_shape_.cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype loss_weight = top[0]->cpu_diff()[0] / outer_num_ * 2;
  Dtype gap_row, gap_col, gap_j;
  for(int i = 0; i < outer_num_; ++i) {
    gap_row = (*mean_row - *(label + 1));
    gap_col = (*mean_col - *label);
    for(int j = 0; j < height_; ++j) {
      gap_j = j - *mean_row;
      for (int k = 0; k < width_; ++k) {
        *(bottom_diff++) = (*(prob_data++)) * (gap_row* gap_j + gap_col*
            (k - *mean_col)) * loss_weight;
      }
    }
    ++mean_row;
    ++mean_col;
    label += 2;
  }
}

#ifdef CPU_ONLY
STUB_GPU(ShapeLossLayer);
#endif

INSTANTIATE_CLASS(ShapeLossLayer);
REGISTER_LAYER_CLASS(ShapeLoss);

}  // namespace caffe
