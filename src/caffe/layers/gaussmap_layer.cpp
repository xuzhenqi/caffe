#include <vector>

#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void GaussMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  std_ = this->layer_param_.gaussmap_param().std();
  width_ = this->layer_param_.gaussmap_param().width();
  height_ = this->layer_param_.gaussmap_param().height();
}


template <typename Dtype>
void GaussMapLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top) {
  vector<int> shape;
  shape.push_back(0);
  shape.push_back(width_);
  shape.push_back(height_);
  for (int i = 0; i < top.size(); ++i) {
    shape[0] = bottom[i]->num();
    top[i]->Reshape(shape);
  }
}

template <typename Dtype>
void GaussMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
  Dtype* top_data;
  const Dtype* bottom_data;
  int encode_loc, loc_width, loc_height;
  for (int i = 0; i < top.size(); ++i) {
    bottom_data = bottom[i]->cpu_data();
    top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < bottom[i]->num(); ++n) {
      encode_loc = int(bottom_data[n] + 0.5);
      loc_width = encode_loc / height_;
      loc_height = encode_loc % height_;
      for (int w = 0; w < width_; ++w) {
        for (int h = 0; h < height_; ++h) {
          //Todo: make exp() correspond to Dtype
          top_data[(n*width_ + w)*height_ + h] = expf(-((w-loc_width)*
              (w-loc_width) + (h-loc_height)*(h-loc_height))/std_/std_);
        }
      }
    }
  }
}

INSTANTIATE_CLASS(GaussMapLayer);
REGISTER_LAYER_CLASS(GaussMap);

} // namespace caffe