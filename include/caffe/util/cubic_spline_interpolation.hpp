#ifndef _CAFFE_UTIL_CUBIC_SPLINE_INTERPOLATION_HPP_
#define _CAFFE_UTIL_CUBIC_SPLINE_INTERPOLATION_HPP_
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace caffe {

template<typename Dtype>
class Func {
 public:
  Dtype a_x, b_x, c_x, d_x;
  Dtype a_y, b_y, c_y, d_y;
  Dtype h;
};

template<typename Dtype>
class CubicSplineInterpolation {
 public:
  CubicSplineInterpolation(const std::vector<cv::Point_<Dtype> >& points)
      : point_v(points) {
    CHECK_GT(point_v.size(), 1);
    if (point_v.size() == 2)
      calc_fun_2pt();
    else
      calc_fun();
  }
  void Interpolation(int nums, std::vector<cv::Point_<Dtype> > &polys,
      bool unifrom = false);
 private:
  void calc_fun();
  void calc_fun_2pt();
  void Uniform(std::vector<cv::Point_<Dtype> >& polys);
  std::vector<cv::Point_<Dtype> > point_v;
  std::vector<Func<Dtype>> func_v;
};

} // namespace caffe

#endif // CAFFE_UTIL_CUBIC_SPLINE_INTERPOLATION_HPP