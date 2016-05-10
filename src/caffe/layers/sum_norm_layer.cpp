#include <iostream>
#include <vector>
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
bool SumNormLayer<Dtype>::ParseRanges(string str, vector<int>& range) {
  range.clear();
  if (str.empty())
    return true;
  str.push_back(',');
  int num = 0, num2 = 0;
  int state = 0;
  for(int i = 0; i < str.size(); ++i) {
    switch(state) {
      case 0:
        if (str[i] >= '0' && str[i] <= '9') {
          num = str[i] - '0';
          state = 1;
        } else
          return false;
        break;
      case 1:
        if (str[i] >= '0' && str[i] <= '9')
          num = num * 10 + str[i] - '0';
        else if (str[i] == ',') {
          range.push_back(num);
          min_channels_ = std::max(num + 1, min_channels_);
          state = 0;
        } else if (str[i] == '-') {
          state = 2;
        } else
          return false;
        break;
      case 2:
        if (str[i] >= '0' && str[i] <= '9') {
          num2 = str[i] - '0';
          state = 3;
        } else
          return false;
        break;
      case 3:
        if (str[i] >= '0' && str[i] <= '9') {
          num2 = num2 * 10 + str[i] - '0';
        } else if (str[i] == ',') {
          if (num2 < num)
            return false;
          for (int r = num; r < num2; ++r)
            range.push_back(r);
          min_channels_ = std::max(min_channels_, num2);
          state = 0;
        } else
          return false;
        break;
    }
  }
  return true;
}

template <typename Dtype>
void SumNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  const SumNormParameter& param = this->layer_param_.sum_norm_param();
  ranges_.resize(param.channels_size());
  prob_ = param.prob();
  CHECK(prob_) << "Only support prob version.";
  for(int i = 0; i < ranges_.size(); ++i) {
    CHECK(ParseRanges(param.channels(i), ranges_[i])) << "Invalid channels "
        "string";
  }
}

template <typename Dtype>
void SumNormLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  CHECK_GE(bottom[0]->channels(), min_channels_);
  vector<int> shape = bottom[0]->shape();
  shape[1] = ranges_.size();
  top[0]->Reshape(shape);
}


template <typename Dtype>
void SumNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  for(int i = 0; i < ranges_.size(); ++i) {
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int j = 0; j < ranges_[i].size(); ++j) {
        caffe_add(bottom[0]->count(2), top_data + top[0]->offset(n, i),
                  bottom_data + bottom[0]->offset(n, ranges_[i][j]),
                  top_data + top[0]->offset(n, i));
      }
      caffe_scal(top[0]->count(2), Dtype(1. / ranges_[i].size()),
                 top_data + top[0]->offset(n, i));
    }
  }
}

template <typename Dtype>
void SumNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                       const vector<bool> &propagate_down,
                                       const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->mutable_cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    for (int i = 0; i < ranges_.size(); ++i) {
      for (int n = 0; n < bottom[0]->num(); ++n) {
        for (int j = 0; j < ranges_[i].size(); ++j) {
          caffe_axpy(top[0]->count(2), Dtype(1./ranges_[i].size()),
                      top_diff + top[0]->offset(n, i),
                      bottom_diff + bottom[0]->offset(n, ranges_[i][j]));
        }
      }
    }
  }
}



#ifdef CPU_ONLY
STUB_GPU(SumNormLayer);
#endif

INSTANTIATE_CLASS(SumNormLayer);
REGISTER_LAYER_CLASS(SumNorm);

} // namespace caffe
