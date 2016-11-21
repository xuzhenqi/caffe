#include <vector>
#include <string>
#include <fstream>
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void AlignmentAccuracyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
  top[1]->Reshape(top_shape);
  top[2]->Reshape(top_shape);
  CHECK_EQ(bottom[0]->count(), 689 * 68 * 2);
  CHECK_EQ(bottom[1]->count(), 689 * 68 * 2);
}

template <typename Dtype>
void AlignmentAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  Dtype comm_err = 0, chal_err = 0;
  const Dtype *bottom_data = bottom[0]->cpu_data();
  const Dtype *bottom_label = bottom[1]->cpu_data();
  const int num = bottom[1]->num();
  Dtype eye_wl, eye_hl, eye_wr, eye_hr, eye_dis;
  for (int n = 0; n < num; ++n) {
    eye_hl = 0; eye_hr = 0; eye_wl = 0; eye_wr = 0;
    for (int c = 36; c < 42; ++c) {
      eye_wl += bottom_label[2*c];
      eye_hl += bottom_label[2*c + 1];
    }
    for (int c = 42; c < 48; ++c) {
      eye_wr += bottom_label[2*c];
      eye_hr += bottom_label[2*c + 1];
    }
    eye_hl /= 6; eye_hr /= 6; eye_wl /= 6; eye_wr /= 6;
    eye_dis = sqrt(square(eye_hl - eye_hr) + square(eye_wl - eye_wr));
    Dtype err = 0;
    for (int c = 0; c < 68; ++c) {
      err += sqrt(square(bottom_data[2*c] - bottom_label[2*c]) + square(
              bottom_data[2*c + 1] - bottom_label[2*c + 1])) / eye_dis;
    }
    if (n < 554) {
      comm_err += err;
    } else {
      chal_err += err;
    }
    bottom_data += 68 * 2;
    bottom_label += 68 * 2;
  }
  top[0]->mutable_cpu_data()[0] = comm_err / 554 / 68 * 100;
  top[1]->mutable_cpu_data()[0] = chal_err / 135 / 68 * 100;
  top[2]->mutable_cpu_data()[0] = (comm_err + chal_err) / num / 68 * 100;
}

INSTANTIATE_CLASS(AlignmentAccuracyLayer);
REGISTER_LAYER_CLASS(AlignmentAccuracy);

}  // namespace caffe
