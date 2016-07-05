#include <vector>

#include "caffe/common_layers.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

namespace caffe {

template <typename Dtype>
void GaussMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  const GaussMapParameter& param = this->layer_param_.gaussmap_param();
  size_ = param.size();
  scale_ = Dtype(size_) / param.ori_size();
  std_ = param.ori_std() * scale_;
}


template <typename Dtype>
void GaussMapLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top) {
  CHECK(bottom[0]->count(1) % 2 == 0);
  vector<int> shape;
  shape.push_back(bottom[0]->num());
  shape.push_back(bottom[0]->count(1) / 2);
  shape.push_back(size_);
  shape.push_back(size_);
  top[0]->Reshape(shape);
}

template <typename Dtype>
void GaussMapLayer<Dtype>::Gauss_map(float x, float y, int height, int width,
                                     Dtype *map) {
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      map[h*width + w] = expf((-(h-y)*(h-y)-(w-x)*(w-x))/std_/std_);
    }
  }
}

template <typename Dtype>
void GaussMapLayer<Dtype>::Regularize(int num, Dtype *map) {
  Dtype sum = 0;
  for (int i = 0; i < num; ++i) {
    sum += map[i];
  }
  for (int i = 0; i < num; ++i) {
    map[i] /= sum;
  }
}

template <typename Dtype>
void GaussMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < bottom[0]->count(); i += 2) {
    Gauss_map(bottom_data[i] * scale_, bottom_data[i+1] * scale_,
              size_, size_, top_data);
    /*
    for (int s1 = 0; s1 < size_; ++s1) {
      for (int s2 = 0; s2 < size_; ++s2) {
        std::cout << top_data[s1 * size_ + s2] << " ";
      }
      std::cout << std::endl;
    }
     */
    Regularize(size_ * size_, top_data);
    /*
    int type = sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1;
    cv::Mat map(size_, size_, type, (void*)top_data), map_resize;
    cv::resize(map, map_resize, cv::Size(1000, 1000));
    cv::imshow("mapin", map_resize);
    cv::waitKey(0);
     */
    top_data += size_ * size_;
  }
}

INSTANTIATE_CLASS(GaussMapLayer);
REGISTER_LAYER_CLASS(GaussMap);

} // namespace caffe