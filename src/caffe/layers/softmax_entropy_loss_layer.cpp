#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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
}

template <typename Dtype>
void SoftmaxEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  for (int i = 0; i < softmax_axis_; ++i) {
    CHECK_EQ(bottom[0]->shape(i), bottom[1]->shape(i));
  }
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
}

template <typename Dtype>
void SoftmaxEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int d = 0; d < dim; ++d) {
      DCHECK_GE(label[i*dim + d], 0);
      DCHECK_LE(label[i*dim + d], 1);
      loss += label[i*dim + d] * (log(std::max(label[i*dim + d], Dtype
          (FLT_MIN))) - log(std::max(prob_data[i*dim + d], Dtype(FLT_MIN))));
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / outer_num_;
}

template <typename Dtype>
void SoftmaxEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  Dtype sum;
  if (propagate_down[0]) {
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype *prob_data = prob_.cpu_data();
    const Dtype *label_data = bottom[1]->cpu_data();
    for (int i = 0; i < prob_.count(); ++i) {
      bottom_diff[i] = prob_data[i] - label_data[i];
    }
    Dtype loss_weight = top[0]->cpu_diff()[0] / outer_num_;
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxEntropyLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxEntropyLossLayer);
REGISTER_LAYER_CLASS(SoftmaxEntropyLoss);

}  // namespace caffe
