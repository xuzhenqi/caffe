#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SumRNNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  int coeff_size = this->layer_param().sum_rnn_param().coeff_size();
  CHECK(coeff_size == 0 || coeff_size == 2);
  coeffs_.resize(2, 0.5);
  if (coeff_size == 2) {
    coeffs_[0] = this->layer_param().sum_rnn_param().coeff(0);
    coeffs_[1] = this->layer_param().sum_rnn_param().coeff(1);
  }
}

template <typename Dtype>
void SumRNNLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {
  CHECK(bottom[0]->shape() == bottom[1]->shape());
  CHECK_EQ(bottom[0]->num(), bottom[2]->count());
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SumRNNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_data_rnn = bottom[1]->cpu_data();
  const Dtype* begin_marker = bottom[2]->cpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = count / num;
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(count, bottom_data, top_data);
  for(int i = 0; i < num; ++i) {
    if (begin_marker[i] <= 0.5) {
      caffe_cpu_axpby(dim, coeffs_[1], bottom_data_rnn, coeffs_[0], top_data);
    }
    bottom_data_rnn += dim;
    top_data += dim;
  }
  bottom_data = bottom[0]->cpu_data();
  bottom_data_rnn = bottom[1]->cpu_data();
  top_data = top[0]->mutable_cpu_data();
}

template <typename Dtype>
void SumRNNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                      const vector<bool> &propagate_down,
                                      const vector<Blob<Dtype> *> &bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_diff_rnn = bottom[1]->mutable_cpu_diff();
  const Dtype* begin_marker = bottom[2]->cpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = count / num;
  const Dtype* top_diff = top[0]->cpu_diff();
  for(int i = 0; i < num; ++i) {
    if (begin_marker[i] <= 0.5) {
      caffe_cpu_scale(dim, coeffs_[0], top_diff, bottom_diff);
      caffe_cpu_scale(dim, coeffs_[1], top_diff, bottom_diff_rnn);
    } else {
      caffe_copy(dim, top_diff, bottom_diff);
      caffe_set(dim, Dtype(0), bottom_diff_rnn);
    }
    bottom_diff += dim;
    bottom_diff_rnn += dim;
    top_diff += dim;
  }
}

#ifdef CPU_ONLY
STUB_GPU(SumRNNLayer);
#endif

INSTANTIATE_CLASS(SumRNNLayer);
REGISTER_LAYER_CLASS(SumRNN);

} // namespace caffe
