#include <vector>
#include "caffe/common.hpp"
#include "glog/logging.h"
#include "caffe/util/cubic_spline_interpolation.hpp"

namespace caffe {

template <typename Dtype>
void CubicSplineInterpolation<Dtype>::calc_fun_2pt() {
  func_v.clear();
  func_v.resize(1);
  func_v[0].h = sqrt(pow(point_v[1].x - point_v[0].x, 2) +
      pow(point_v[1].y - point_v[0].y, 2));
  CHECK_GT(func_v[0].h, 1e-6); // h[i] should not be 0, or the consequent points
  // should not be equal. To avoid numerical error, better set h[i] to 1e-6
  // (or other values bigger than 0.) if h[i] is close to 0.
  func_v[0].a_x = point_v[0].x;
  func_v[0].b_x = (point_v[1].x - point_v[0].x) / func_v[0].h;
  func_v[0].c_x = 0;
  func_v[0].d_x = 0;
  func_v[0].a_y = point_v[0].y;
  func_v[0].b_y = (point_v[1].y - point_v[0].y) / func_v[0].h;
  func_v[0].c_x = 0;
  func_v[0].d_x = 0;
}

template <typename Dtype>
void CubicSplineInterpolation<Dtype>::calc_fun() {

  int n = point_v.size();
  std::vector<Dtype> Mx(n);
  std::vector<Dtype> My(n);
  std::vector<Dtype> A(n-2);
  std::vector<Dtype> B(n-2);
  std::vector<Dtype> C(n-2);
  std::vector<Dtype> Dx(n-2);
  std::vector<Dtype> Dy(n-2);
  std::vector<Dtype> h(n-1);
  func_v.clear();
  func_v.resize(n-1);
  //std::vector<func> func_v(n-1);

  for(int i = 0; i < n-1; i++)
  {
    h[i] = sqrt(pow(point_v[i+1].x - point_v[i].x, 2) + pow(point_v[i+1].y - point_v[i].y, 2));
    CHECK_GT(h[i], 1e-6); // h[i] should not be 0, or the consequent points
    // should not be equal. To avoid numerical error, better set h[i] to 1e-6
    // (or other values bigger than 0.) if h[i] is close to 0.
  }

  for(int i = 0; i < n-2; i++)
  {
    A[i] = h[i];
    B[i] = 2*(h[i]+h[i+1]);
    C[i] = h[i+1];

    Dx[i] =  6*( (point_v[i+2].x - point_v[i+1].x)/h[i+1] - (point_v[i+1].x - point_v[i].x)/h[i] );
    Dy[i] =  6*( (point_v[i+2].y - point_v[i+1].y)/h[i+1] - (point_v[i+1].y - point_v[i].y)/h[i] );
  }

  //TDMA
  C[0] = C[0] / B[0];
  Dx[0] = Dx[0] / B[0];
  Dy[0] = Dy[0] / B[0];
  for(int i = 1; i < n-2; i++)
  {
    Dtype tmp = B[i] - A[i]*C[i-1];
    C[i] = C[i] / tmp;
    Dx[i] = (Dx[i] - A[i]*Dx[i-1]) / tmp;
    Dy[i] = (Dy[i] - A[i]*Dy[i-1]) / tmp;
  }
  Mx[n-2] = Dx[n-3];
  My[n-2] = Dy[n-3];
  for(int i = n-4; i >= 0; i--)
  {
    Mx[i+1] = Dx[i] - C[i]*Mx[i+2];
    My[i+1] = Dy[i] - C[i]*My[i+2];
  }

  Mx[0] = 0;
  Mx[n-1] = 0;
  My[0] = 0;
  My[n-1] = 0;

  for(int i = 0; i < n-1; i++)
  {
    func_v[i].a_x = point_v[i].x;
    func_v[i].b_x = (point_v[i+1].x - point_v[i].x)/h[i] - (2*h[i]*Mx[i] + h[i]*Mx[i+1]) / 6;
    func_v[i].c_x = Mx[i]/2;
    func_v[i].d_x = (Mx[i+1] - Mx[i]) / (6*h[i]);

    func_v[i].a_y = point_v[i].y;
    func_v[i].b_y = (point_v[i+1].y - point_v[i].y)/h[i] - (2*h[i]*My[i] + h[i]*My[i+1]) / 6;
    func_v[i].c_y = My[i]/2;
    func_v[i].d_y = (My[i+1] - My[i]) / (6*h[i]);

    func_v[i].h = h[i];
  }
}


template <typename Dtype>
void CubicSplineInterpolation<Dtype>::Interpolation(
    int nums, std::vector<cv::Point_<Dtype>> &polys, bool uniform) {
  polys.clear();
  for(int j = 0; j < func_v.size(); j++)
  {
    Dtype delta = func_v[j].h / nums;
    for(int k = 0; k < nums; ++k)
    {
      Dtype t1 = delta*k;
      Dtype x1 = func_v[j].a_x + func_v[j].b_x*t1 + func_v[j].c_x*pow(t1,2) +
          func_v[j].d_x*pow(t1,3);
      Dtype y1 = func_v[j].a_y + func_v[j].b_y*t1 + func_v[j].c_y*pow(t1,2) +
          func_v[j].d_y*pow(t1,3);

      polys.push_back(cv::Point_<Dtype>(x1,y1));
    }
  }
  if (uniform) {
    Uniform(polys);
  }
}

/*
 * Uniform: re-interpolate the points of polys to make it distributed evenly
 * in the curve, that it make the distance of consequent points equal.
 */
template <typename Dtype>
void CubicSplineInterpolation<Dtype>::Uniform(
    std::vector<cv::Point_<Dtype> > &polys) {
  const int nums = polys.size();
  if (nums <= 2)
    return;
  vector<Dtype> length(nums, 0);
  Dtype dx, dy;
  for (int i = 1; i < nums; ++i) {
    dx = polys[i].x - polys[i-1].x;
    dy = polys[i].y - polys[i-1].y;
    length[i] = length[i - 1] + sqrt(dx * dx + dy * dy);
  }
  Dtype delta_length = length.back() / (nums - 1), cur_length = 0;
  int cur_index = 0, ratio;
  vector<cv::Point_<Dtype> > polys_buf(polys.begin(), polys.end());

  for (int i = 1; i < nums - 1; ++i) {
    cur_length = delta_length * i;
    while(cur_length > length[cur_index + 1]) {
      ++cur_index;
    }
    ratio = (cur_length - length[cur_index]) / (length[cur_index + 1] -
        length[cur_index]);
    polys[i].x = ratio * (polys_buf[cur_index + 1].x - polys_buf[cur_index].x)
        + polys_buf[cur_index].x;
    polys[i].y = ratio * (polys_buf[cur_index + 1].y - polys_buf[cur_index].y)
        + polys_buf[cur_index].y;
  }
}

INSTANTIATE_CLASS(Func);
INSTANTIATE_CLASS(CubicSplineInterpolation);

} // namespace caffe

