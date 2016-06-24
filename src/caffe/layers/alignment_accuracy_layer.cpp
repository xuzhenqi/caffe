#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/cubic_spline_interpolation.hpp"
#include "caffe/loss_layers.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

namespace caffe {
  
/**
* @brief accuracy layer for face alignment @f$
*
*/

template <typename Dtype>
void AlignmentAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  AlignmentAccuracyParameter
      alignment_accuracy_param = this->layer_param_.alignment_accuracy_param();

  //check
  CHECK(bottom[0]->num() == bottom[1]->num()
            && bottom[0]->num() == bottom[2]->num())
  << "\n2 bottoms should be at the same number ";
  CHECK(((bottom[1]->channels() > 1) ? 1 : 0 + (bottom[1]->height() > 1) ? 1 : 0
                                                                                   + (bottom[1]->width()
                                                                                       > 1)
                                                                               ? 1
                                                                               : 0)
            == 1)
  << "\ninput result must be a vector ";
  CHECK(((bottom[2]->channels() > 1) ? 1 : 0 + (bottom[2]->height() > 1) ? 1 : 0
                                                                                   + (bottom[2]->width()
                                                                                       > 1)
                                                                               ? 1
                                                                               : 0)
            == 1)
  << "\ninput label must be a vector ";
  CHECK(bottom[2]->count(1) % 2 == 0)
  << "\ninput result's length must be even ";
  CHECK(bottom[2]->count(1) % 2 == 0)
  << "\ninput label's length must be even ";
  CHECK(bottom[1]->count(1) == bottom[2]->count(1))
  << "\ninput label's length and result's length must be equal ";
  CHECK(alignment_accuracy_param.scale() > 0)
  << "\nscale must be positive";

  if (alignment_accuracy_param.use_mean_pose()) {
    CHECK(alignment_accuracy_param.landmark_x_size()
              == alignment_accuracy_param.landmark_y_size())
    << "\nlandmark_x's number must equal to landmark_y's number ";
    CHECK(bottom[1]->count(1) == alignment_accuracy_param.landmark_x_size() * 2)
    << "\ninput points' number must be equal to landmarks' number ";
  }

  //assign
  seperate_accuracy = alignment_accuracy_param.seperate_accuracy();
  point_num = bottom[1]->count(1) / 2;
  output_num = alignment_accuracy_param.output_num() > bottom[0]->num()
               ? bottom[0]->num() : alignment_accuracy_param.output_num();
  output_threshold = alignment_accuracy_param.output_threshold();
  scale = alignment_accuracy_param.scale();
  use_mean_pose = alignment_accuracy_param.use_mean_pose();
  for (size_t i = 0; i < alignment_accuracy_param.left_eye_index_size(); i++) {
    left_eye_index.push_back(alignment_accuracy_param.left_eye_index(i));
  }
  for (size_t i = 0; i < alignment_accuracy_param.right_eye_index_size(); i++) {
    right_eye_index.push_back(alignment_accuracy_param.right_eye_index(i));
  }

  if (use_mean_pose) {
    Point_<Dtype> left_eye(0, 0);
    Point_<Dtype> right_eye(0, 0);
    for (size_t i = 0; i < point_num; i++) {
      mean_pose.push_back(Point_<Dtype>(alignment_accuracy_param.landmark_x(i),
                                  alignment_accuracy_param.landmark_y(i)));
    }
    for (size_t i = 0; i < left_eye_index.size(); i++) {
      left_eye += mean_pose[left_eye_index[i]];
    }
    left_eye = left_eye * (1.0 / left_eye_index.size());
    for (size_t i = 0; i < right_eye_index.size(); i++) {
      right_eye += mean_pose[right_eye_index[i]];
    }
    right_eye = right_eye * (1.0 / right_eye_index.size());
    norm_factor_static = sqrt(
        (left_eye.x - right_eye.x) * (left_eye.x - right_eye.x)
            + (left_eye.y - right_eye.y) * (left_eye.y - right_eye.y));
  }
  if (alignment_accuracy_param.has_output_image_prefix() && output_num > 0) {
    output_image = true;
    output_image_prefix = alignment_accuracy_param.output_image_prefix();
  }
  else {
    output_image = false;
    output_image_prefix = "";
  }
  if (alignment_accuracy_param.has_output_error_file()) {
    output_error = true;
    output_error_file = alignment_accuracy_param.output_error_file();
  }
  else {
    output_error = false;
    output_error_file = "";
  }

  batch_counter = 0;

  // SetUp area loss related members
  point_loss_.resize(point_num, true);
  interpolation_nums_ = alignment_accuracy_param.interpolation_nums();
  if (interpolation_nums_ != 0) {
    CHECK_EQ(alignment_accuracy_param.begin_size(),
             alignment_accuracy_param.end_size());
    ranges_.resize(alignment_accuracy_param.begin_size());
    for (int i = 0; i < alignment_accuracy_param.begin_size(); ++i) {
      ranges_[i].push_back(alignment_accuracy_param.begin(i));
      ranges_[i].push_back(alignment_accuracy_param.end(i));
      CHECK_GE(ranges_[i][0], 0);
      CHECK_GE(ranges_[i][1] - ranges_[i][0], 2);
      CHECK_LT(ranges_[i][1], point_num);
      for (int j = ranges_[i][0]; j < ranges_[i][1]; ++j) {
        point_loss_[j] = false;
      }
    }
  }
}

template <typename Dtype>
void AlignmentAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  if(seperate_accuracy)
  {
    vector<int> top_shape(1);
    top_shape[0]=point_num;
    top[0]->Reshape(top_shape);
  }
  else
  {
    vector<int> top_shape(0);
    top[0]->Reshape(top_shape);
  }
  if(top.size()==2)
  {
    vector<int> top_shape(1);
    top_shape[0]=bottom[0]->num();
    top[1]->Reshape(top_shape);
  }
}

template <typename Dtype>
Dtype AlignmentAccuracyLayer<Dtype>::compute_inter_pupil_dis(
    const vector<cv::Point_<Dtype>> &point_label) {
  Dtype inter_pupil_dis;
  if (use_mean_pose) {
    Mat X(2 * point_num, 4,
          (sizeof(Dtype) == sizeof(float)) ? CV_32F : CV_64F);
    Mat U(2 * point_num, 1,
          (sizeof(Dtype) == sizeof(float)) ? CV_32F : CV_64F);
    for (unsigned int i = 0; i < point_num; i++) {
      X.at<Dtype>(i, 0) = point_label[i].x;
      X.at<Dtype>(i + point_num, 0) = point_label[i].y;
      X.at<Dtype>(i, 1) = point_label[i].y;
      X.at<Dtype>(i + point_num, 1) = -1 * point_label[i].x;
      X.at<Dtype>(i, 2) = 1;
      X.at<Dtype>(i + point_num, 3) = 1;
      X.at<Dtype>(i, 3) = 0;
      X.at<Dtype>(i + point_num, 2) = 0;

      U.at<Dtype>(i, 0) = mean_pose[i].x;
      U.at<Dtype>(i + point_num, 0) = mean_pose[i].y;
    }
    Mat result = X.inv(DECOMP_SVD) * U;
    Dtype mean_pose_scale = sqrt(
        result.at<Dtype>(0, 0) * result.at<Dtype>(0, 0)
            + result.at<Dtype>(1, 0) * result.at<Dtype>(1, 0));
    inter_pupil_dis = norm_factor_static / mean_pose_scale;
  }
  else {
    Point_<Dtype> left_eye(0, 0);
    Point_<Dtype> right_eye(0, 0);
    for (size_t i = 0; i < left_eye_index.size(); i++) {
      left_eye += point_label[left_eye_index[i]];
    }
    left_eye = left_eye * (1.0 / left_eye_index.size());
    for (size_t i = 0; i < right_eye_index.size(); i++) {
      right_eye += point_label[right_eye_index[i]];
    }
    right_eye = right_eye * (1.0 / right_eye_index.size());
    inter_pupil_dis = sqrt(
        (left_eye.x - right_eye.x) * (left_eye.x - right_eye.x)
            + (left_eye.y - right_eye.y) * (left_eye.y - right_eye.y));
  }
  return inter_pupil_dis;
}

template <typename Dtype>
Dtype AlignmentAccuracyLayer<Dtype>::compute_area_loss(
    const vector<cv::Point_<Dtype>> &point_result,
    const vector<cv::Point_<Dtype>> &point_label,
    int begin, int end, Dtype inter_pupil_dis) {
  vector<cv::Point_<Dtype> > result(point_result.begin() + begin,
      point_result.begin() + end);
  vector<cv::Point_<Dtype> > label(point_label.begin() + begin,
      point_label.begin() + end);
  CubicSplineInterpolation<Dtype> spline_result(result), spline_label(label);
  vector<cv::Point_<Dtype> > inter_result, inter_label;
  spline_result.Interpolation(interpolation_nums_, inter_result, true);
  spline_label.Interpolation(interpolation_nums_, inter_label, true);
  Dtype err = 0, dx, dy;
  for (int i = 0; i < inter_label.size(); ++i) {
    dx = inter_result[i].x - inter_label[i].x;
    dy = inter_result[i].y - inter_label[i].y;
    err += sqrt(dx * dx + dy * dy);
  }
  return err / inter_pupil_dis / inter_label.size();
}

template <typename Dtype>
Dtype AlignmentAccuracyLayer<Dtype>::compute_err_each_point(
    const vector<cv::Point_<Dtype>> &point_result,
    const vector<cv::Point_<Dtype>> &point_label,
    const Dtype inter_pupil_dis,
    vector<Dtype> &error_each_point_num) {
  Dtype err = 0;
  for (unsigned int i = 0; i < point_num; i++) {
    if (point_loss_[i]) {
      error_each_point_num[i] = sqrt(
          (point_result[i].x - point_label[i].x) *
              (point_result[i].x - point_label[i].x) +
              (point_result[i].y - point_label[i].y) *
                  (point_result[i].y - point_label[i].y)) / inter_pupil_dis;
      err += error_each_point_num[i] / point_num;
    }
  }
  for (unsigned int j = 0; j < ranges_.size(); ++j) {
    Dtype area_loss = compute_area_loss(point_result, point_label,
                                        ranges_[j][0], ranges_[j][1],
                                        inter_pupil_dis);
    for (unsigned int i = ranges_[j][0]; i < ranges_[j][1]; ++i){
      error_each_point_num[i] = area_loss;
    }
    err += area_loss * (ranges_[j][1] - ranges_[j][0]) / point_num;
  }
  return err;
}

template <typename Dtype>
void AlignmentAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bottom_image=bottom[0]->cpu_data();
  const Dtype* bottom_result=bottom[1]->cpu_data();
  const Dtype* bottom_label=bottom[2]->cpu_data(); 

  //calculate error for each image
  vector<Dtype> error(bottom[0]->num(), 0);
  vector<vector<Dtype> > error_each_point(bottom[0]->num());
  for(size_t i=0;i<bottom[0]->num();i++)
  {
    error_each_point[i].resize(point_num, 0);
  }

  for(unsigned int num=0;num<bottom[0]->num();num++) {
    size_t bottom_result_offset = bottom[1]->offset(num);
    size_t bottom_label_offset = bottom[2]->offset(num);

    vector<Point_<Dtype>> point_result(point_num);
    vector<Point_<Dtype>> point_label(point_num);
    for (unsigned int i = 0; i < point_num; i++) {
      point_result[i].x = bottom_result[i * 2 + bottom_result_offset];
      point_result[i].y = bottom_result[i * 2 + 1 + bottom_result_offset];
      point_label[i].x = bottom_label[i * 2 + bottom_label_offset];
      point_label[i].y = bottom_label[i * 2 + 1 + bottom_label_offset];
    }

    Dtype inter_pupil_dis = compute_inter_pupil_dis(point_label);

    error[num] = compute_err_each_point(point_result, point_label,
                                        inter_pupil_dis,
                                        error_each_point[num]);
  }
  
  //output error in each batch
  if(output_error)
  {
    FILE *fp_error=fopen(output_error_file.c_str(), "a");
    for(size_t i=0;i<error.size();i++)
    {
      fprintf(fp_error, "%f\n", error[i]);
    }
    fclose(fp_error);
  }
  
  if(output_image)
  {
    //sort
    vector<unsigned int> order(bottom[0]->num());
    for(unsigned int num=0;num<bottom[0]->num();num++)
    {
      order[num]=num;
    }
    for(unsigned int i=0;i<bottom[0]->num();i++)
    {
      for(unsigned int j=i+1;j<bottom[0]->num();j++)
      {
        if(error[order[j]]>error[order[i]])
        {
          unsigned int temp=order[j];
          order[j]=order[i];
          order[i]=temp;
        }
      }
    }
    
    //output some image
    for(unsigned int num=0;num<output_num;num++)
    {
      if(error[order[num]]<output_threshold)
      {
        break;  
      }
      
      //landmark
      size_t bottom_result_offset=bottom[1]->offset(order[num]);
  		size_t bottom_label_offset=bottom[2]->offset(order[num]);
  
  		vector<Point_<Dtype>> point_result(point_num);
      vector<Point_<Dtype>> point_label(point_num);
  		for(unsigned int i=0;i<point_num;i++)
  		{
  			point_result[i].x=bottom_result[i*2+bottom_result_offset];
  			point_result[i].y=bottom_result[i*2+1+bottom_result_offset];
        point_label[i].x=bottom_label[i*2+bottom_label_offset];
  			point_label[i].y=bottom_label[i*2+1+bottom_label_offset];
      }
      
      //image
  		Mat image_dtype(bottom[0]->height(), bottom[0]->width(), (sizeof(Dtype)==sizeof(float))?CV_32F:CV_64F);
  		memcpy(image_dtype.data, bottom_image+bottom[0]->offset(order[num]), sizeof(Dtype)*image_dtype.cols*image_dtype.rows);
  		Dtype min_val=FLT_MAX;
  		Dtype max_val=-1*FLT_MAX;
  		for(unsigned int i=0;i<image_dtype.rows;i++)
  		{
  			for(unsigned int j=0;j<image_dtype.cols;j++)
  			{
  				if(image_dtype.at<Dtype>(i, j)>max_val)
  				{
  					max_val=image_dtype.at<Dtype>(i, j);
  				}
  				if(image_dtype.at<Dtype>(i, j)<min_val)
  				{
  					min_val=image_dtype.at<Dtype>(i, j);
  				}
  			}
  		}
  		Mat iamge_to_show(image_dtype.rows, image_dtype.cols, CV_8U);
  		for(unsigned int i=0;i<image_dtype.rows;i++)
  		{
  			for(unsigned int j=0;j<image_dtype.cols;j++)
  			{
  				iamge_to_show.at<unsigned char>(i, j)=(image_dtype.at<Dtype>(i, j)-min_val)*255/(max_val-min_val);
  			}
  		}
      cvtColor(iamge_to_show, iamge_to_show, CV_GRAY2BGR);
      resize(iamge_to_show, iamge_to_show, Size((int)(iamge_to_show.cols*scale), (int)(iamge_to_show.rows*scale)));
      
      //draw point
      float r=4;
      for(unsigned int i=0;i<point_num;i++)
      {
        circle(iamge_to_show, point_result[i]*scale, r-ceil(r/2), Scalar(0, 0, 255), ceil(r/2));
        circle(iamge_to_show, point_label[i]*scale, r-ceil(r/2), Scalar(0, 255, 0), ceil(r/2));
      }

      char buf[100];
      sprintf(buf, "batch_%u_NO_%u_Raw_%u_Err_%.3f.png", batch_counter, num, batch_counter*bottom[0]->num()+order[num], error[order[num]]*100);
  		imwrite((output_image_prefix+buf).c_str(), iamge_to_show);
    }
  }
  
  //accuracy
  if(seperate_accuracy)
  {
    vector<Dtype> accuracy(point_num, 0);
    for(unsigned int num=0;num<bottom[0]->num();num++)
    {
      for(size_t i=0;i<point_num;i++)
      {
        accuracy[i]+=error_each_point[num][i];
      }
    }
    for(size_t i=0;i<point_num;i++)
    {
      top[0]->mutable_cpu_data()[i]=accuracy[i]/bottom[0]->num();
    }
  }
  else
  {
    float accuracy=0;
    for(unsigned int num=0;num<bottom[0]->num();num++)
    {
      accuracy+=error[num];
    }
    top[0]->mutable_cpu_data()[0]=accuracy/bottom[0]->num();
  }
  
  if(top.size()==2)
  {
    caffe_copy(error.size(), &(error[0]), top[1]->mutable_cpu_data());
  }
  
  batch_counter++;
}

INSTANTIATE_CLASS(AlignmentAccuracyLayer);
REGISTER_LAYER_CLASS(AlignmentAccuracy);

}  // namespace caffe
