#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/loss_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KLDivergenceLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  const KLDivergenceLossParameter &param =
      this->layer_param_.kl_divergence_loss_param();
  margin_ = param.margin();
  axis_ = param.axis();
  CHECK_GE(axis_, 0);
}

template <typename Dtype>
void KLDivergenceLossLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  buffer_.ReshapeLike(*bottom[0]);
  outer_num_ = bottom[0]->count(0, axis_);
}

template <typename Dtype>
void KLDivergenceLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype* pred = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < bottom[0]->count(); ++i) {
    //CHECK(label[i] >= 0 && label[i] <= 1);
    //CHECK(pred[i] >= 0 && pred[i] <= 1);
    loss += label[i] * (log((label[i] + margin_) / (pred[i] + margin_)));
  }
  top[0]->mutable_cpu_data()[0] = loss / outer_num_;
}

template <typename Dtype>
void KLDivergenceLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype *pred_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* pred = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    for (int i = 0; i < bottom[0]->count(); ++i) {
      pred_diff[i] = - label[i] / (pred[i] + margin_);
    }
    Dtype loss_weight = top[0]->cpu_diff()[0] / outer_num_;
    caffe_scal(bottom[0]->count(), loss_weight, pred_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(KLDivergenceLossLayer);
#endif

INSTANTIATE_CLASS(KLDivergenceLossLayer);
REGISTER_LAYER_CLASS(KLDivergenceLoss);

}  // namespace caffe
