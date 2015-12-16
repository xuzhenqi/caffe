#include <functional>
#include <vector>

#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void FaceDetectionAccuracyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
  CHECK_EQ(bottom[0]->channels(), 68);
}

template <typename Dtype>
void FaceDetectionAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  Dtype err = 0;
  const Dtype *bottom_data = bottom[0]->cpu_data();
  const Dtype *bottom_label = bottom[1]->cpu_data();
  const int num = bottom[1]->num();
  const int channel = bottom[1]->channels();
  const int height = bottom[1]->height();
  const int width = bottom[1]->width();
  vector<int> index_data(channel*2, 0), index_label(channel*2, 0);
  Dtype max_data, max_label, eye_dis;
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channel; ++c) {
      index_data[c*2] = 0;
      index_data[c*2+1] = 0;
      index_label[c*2] = 0;
      index_label[c*2+1] = 0;
      max_data = *bottom_data;
      max_label = *bottom_label;
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          if (max_data < *bottom_data) {
            max_data = *bottom_data;
            index_data[2*c] = h;
            index_data[2*c+1] = w;
          }
          if (max_label < *bottom_label) {
            max_label = *bottom_label;
            index_label[2*c] = h;
            index_label[2*c+1] = w;
          }
          ++bottom_data;
          ++bottom_label;
        }
      }
    }
    eye_dis = sqrt(square(index_label[2*36]-index_label[2*45]) + square
        (index_label[2*36+1]-index_label[2*45+1]));
    for (int c = 0; c < channel; ++c) {
      err += sqrt(square(index_data[2*c]-index_label[2*c]) + square
          (index_data[2*c+1]-index_label[2*c+1])) / eye_dis;
    }
  }
  top[0]->mutable_cpu_data()[0] = err / num;
}

INSTANTIATE_CLASS(FaceDetectionAccuracyLayer);
REGISTER_LAYER_CLASS(FaceDetectionAccuracy);

} // namespace caffe
