#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>

#include "caffe/aug_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using namespace std;
using namespace cv;

namespace caffe {

template<typename Dtype>
AugDataTransformer<Dtype>::AugDataTransformer(const AugTransformationParameter& param, Phase phase) 
  : param_(param), phase_(phase) {

    if(param_.crop_h() > 0 && param_.crop_w() > 0) {
        crop_h = param_.crop_h();
        crop_w = param_.crop_w();
        need_crop=true;
    }
    else if(param_.crop_size() > 0) {
        crop_h = param_.crop_size();
        crop_w = param_.crop_size();
        need_crop=true;
    }
    else {
        need_crop=false;
    }

    label_type=param_.label_type();
    has_weight=param_.has_weight();
    aug_each_channel=param_.aug_each_channel();
    
    if(aug_each_channel) {
        CHECK_EQ(label_type, AugTransformationParameter_LabelType_CLASS);
    }

    if(param_.zoom()) {
        int zoom_scale_sources=0;
        zoom_scale_sources+=param_.has_zoom_scale()?1:0;
        zoom_scale_sources+=param_.has_zoom_rng()?1:0;
        CHECK_EQ(zoom_scale_sources, 1);
    }
    if(param_.trans()) {
        int trans_x_sources=0;
        trans_x_sources+=param_.has_trans_x()?1:0;
        trans_x_sources+=param_.has_trans_x_rng()?1:0;
        trans_x_sources+=param_.has_trans_rng()?1:0;
        CHECK_EQ(trans_x_sources, 1);
        int trans_y_sources=0;
        trans_y_sources+=param_.has_trans_y()?1:0;
        trans_y_sources+=param_.has_trans_y_rng()?1:0;
        trans_y_sources+=param_.has_trans_rng()?1:0;
        CHECK_EQ(trans_y_sources, 1);
    }
    if(param_.rotate()) {
        int rotate_deg_sources=0;
        rotate_deg_sources+=param_.has_rotate_deg()?1:0;
        rotate_deg_sources+=param_.has_rotate_rng()?1:0;
        CHECK_EQ(rotate_deg_sources, 1);
    }
    if(param_.occlusion()) {
        int occlusion_center_x_sources=0;
        occlusion_center_x_sources+=param_.has_occlusion_center_x()?1:0;
        occlusion_center_x_sources+=param_.has_occlusion_center_x_rng()?1:0;
        occlusion_center_x_sources+=param_.has_occlusion_center_rng()?1:0;
        CHECK_EQ(occlusion_center_x_sources, 1);
        int occlusion_center_y_sources=0;
        occlusion_center_y_sources+=param_.has_occlusion_center_y()?1:0;
        occlusion_center_y_sources+=param_.has_occlusion_center_y_rng()?1:0;
        occlusion_center_y_sources+=param_.has_occlusion_center_rng()?1:0;
        CHECK_EQ(occlusion_center_y_sources, 1);
        int occlusion_width_sources=0;
        occlusion_width_sources+=param_.has_occlusion_width()?1:0;
        occlusion_width_sources+=param_.has_occlusion_width_rng()?1:0;
        occlusion_width_sources+=param_.has_occlusion_size_rng()?1:0;
        CHECK_EQ(occlusion_width_sources, 1);
        int occlusion_height_sources=0;
        occlusion_height_sources+=param_.has_occlusion_height()?1:0;
        occlusion_height_sources+=param_.has_occlusion_height_rng()?1:0;
        occlusion_height_sources+=param_.has_occlusion_size_rng()?1:0;
        CHECK_EQ(occlusion_height_sources, 1);
        int occlusion_color_sources=0;
        occlusion_color_sources+=param_.has_occlusion_color()?1:0;
        occlusion_color_sources+=param_.has_occlusion_color_rng()?1:0;
        CHECK_EQ(occlusion_color_sources, 1);
    }

    if(param_.corr_list_size()>0) {
        CHECK_EQ(label_type, AugTransformationParameter_LabelType_POINT)
        << "corr_list is only available for label_type=POINT";
    }
    CHECK_EQ(param_.corr_list_size()%2, 0)
    << "\n number of corr_list must be even";
    corr_list.resize(param_.corr_list_size());
    for(int i=0;i<corr_list.size();i++) {
        corr_list[i]=param_.corr_list(i);
    }
}

template<typename Dtype>
void AugDataTransformer<Dtype>::AffinePoint(Point2f& point, const Mat& affine_mat) {
    CHECK_EQ(affine_mat.rows, 2);
    CHECK_EQ(affine_mat.cols, 3);
    CHECK_EQ(affine_mat.channels(), 1);
    CHECK_EQ(affine_mat.depth(), CV_64F);
    
    Point2f result;
    result.x = affine_mat.at<double>(0, 0)*point.x+
                affine_mat.at<double>(0, 1)*point.y+
                affine_mat.at<double>(0, 2);
    result.y = affine_mat.at<double>(1, 0)*point.x+
                affine_mat.at<double>(1, 1)*point.y+
                affine_mat.at<double>(1, 2);
    
    point=result;
}

template<typename Dtype>
void AugDataTransformer<Dtype>::AffineRect(Point2f& left_top, Point2f& right_bottom, const Mat& affine_mat) {
    CHECK_EQ(affine_mat.rows, 2);
    CHECK_EQ(affine_mat.cols, 3);
    CHECK_EQ(affine_mat.channels(), 1);
    CHECK_EQ(affine_mat.depth(), CV_64F);
    
    CHECK_LE(left_top.x, right_bottom.x);
    CHECK_LE(left_top.y, right_bottom.y);
    
    Point2f affined_left_top(left_top.x, left_top.y);
    Point2f affined_left_bottom(left_top.x, right_bottom.y);
    Point2f affined_right_top(right_bottom.x, left_top.y);
    Point2f affined_right_bottom(right_bottom.x, right_bottom.y);
    
    AffinePoint(affined_left_top, affine_mat);
    AffinePoint(affined_left_bottom, affine_mat);
    AffinePoint(affined_right_top, affine_mat);
    AffinePoint(affined_right_bottom, affine_mat);
    
    const Point2f affined_center((affined_left_top.x+affined_right_bottom.x)/2, (affined_left_top.y+affined_right_bottom.y)/2);
    
    const float line_left_a=affined_left_top.y-affined_left_bottom.y;
    const float line_left_b=affined_left_bottom.x-affined_left_top.x;
    const float line_left_c=affined_left_top.x*affined_left_bottom.y-affined_left_bottom.x*affined_left_top.y;
    const float line_top_a=affined_right_top.y-affined_left_top.y;
    const float line_top_b=affined_left_top.x-affined_right_top.x;
    const float line_top_c=affined_right_top.x*affined_left_top.y-affined_left_top.x*affined_right_top.y;
    const float line_right_a=affined_right_bottom.y-affined_right_top.y;
    const float line_right_b=affined_right_top.x-affined_right_bottom.x;
    const float line_right_c=affined_right_bottom.x*affined_right_top.y-affined_right_top.x*affined_right_bottom.y;
    const float line_bottom_a=affined_left_bottom.y-affined_right_bottom.y;
    const float line_bottom_b=affined_right_bottom.x-affined_left_bottom.x;
    const float line_bottom_c=affined_left_bottom.x*affined_right_bottom.y-affined_right_bottom.x*affined_left_bottom.y;
    
    vector<float> x_cross(4);
    x_cross[0]=-1*(line_left_b*affined_center.y+line_left_c)/line_left_a;
    x_cross[1]=-1*(line_top_b*affined_center.y+line_top_c)/line_top_a;
    x_cross[2]=-1*(line_right_b*affined_center.y+line_right_c)/line_right_a;
    x_cross[3]=-1*(line_bottom_b*affined_center.y+line_bottom_c)/line_bottom_a;
    
    vector<float> y_cross(4);
    y_cross[0]=-1*(line_left_a*affined_center.x+line_left_c)/line_left_b;
    y_cross[1]=-1*(line_top_a*affined_center.x+line_top_c)/line_top_b;
    y_cross[2]=-1*(line_right_a*affined_center.x+line_right_c)/line_right_b;
    y_cross[3]=-1*(line_bottom_a*affined_center.x+line_bottom_c)/line_bottom_b;
    
    sort(x_cross.begin(), x_cross.end(), less<float>());
    sort(y_cross.begin(), y_cross.end(), less<float>());
    
    // if the affined rect with rotate has rotate-degree=0, there will be 2 inf in x_cross and y_cross
    vector<float> x_corss_valid;
    for(size_t i=0;i<x_cross.size();i++) {
        if(!isinf(x_cross[i])) {
            x_corss_valid.push_back(x_cross[i]);
        }
    }
    vector<float> y_corss_valid;
    for(size_t i=0;i<y_cross.size();i++) {
        if(!isinf(y_cross[i])) {
            y_corss_valid.push_back(y_cross[i]);
        }
    }
    CHECK(x_corss_valid.size()==2 || x_corss_valid.size()==4);
    CHECK(y_corss_valid.size()==2 || y_corss_valid.size()==4);
    size_t start_x_index=(x_corss_valid.size()-2)/2;
    size_t start_y_index=(y_corss_valid.size()-2)/2;
    left_top.x=x_corss_valid[start_x_index];
    left_top.y=y_corss_valid[start_y_index];
    right_bottom.x=x_corss_valid[start_x_index+1];
    right_bottom.y=y_corss_valid[start_y_index+1];
}

template<typename Dtype>
void AugDataTransformer<Dtype>::WarpAffineMultiChannels(const cv::Mat& input, cv::Mat& output,
        const cv::Mat& trans_mat, const cv::Size& output_size, int interp_type, int border_type, const cv::Scalar& border_value) {
    if(input.channels()<=4) {
        warpAffine(input, output, trans_mat, output_size, interp_type, border_type, border_value);
    }
    else {
        vector<Mat> split_input(input.channels());
        vector<Mat> split_output(input.channels());
        split(input, split_input.data());
        for(size_t i=0;i<split_input.size();i+=4) {
            size_t need_trans_channels=std::min(static_cast<size_t>(4), split_input.size()-i);
            Mat input_4;
            Mat output_4;
            vector<Mat> split_input_4(need_trans_channels);
            vector<Mat> split_output_4(need_trans_channels);
            for(size_t j=0;j<split_input_4.size();j++) {
                split_input_4[j]=split_input[i+j];
            }
            merge(split_input_4, input_4);
            warpAffine(input_4, output_4, trans_mat, output_size, interp_type, border_type, border_value);
            split(output_4, split_output_4.data());
            for(size_t j=0;j<split_output_4.size();j++) {
                split_output[i+j]=split_output_4[j];
            }
        }
        merge(split_output, output);
    }
}

template<typename Dtype>
void AugDataTransformer<Dtype>::Transform(const cv::Mat& img,
                                          const cv::Mat& label,
                                          Blob<Dtype>& transformed_img,
                                          Blob<Dtype>& transformed_label) {
                                            
    const int input_img_channels = img.channels();
    const int input_img_height = img.rows;
    const int input_img_width = img.cols;

    CHECK_EQ(img.depth(), CV_8U);

    if(label_type==AugTransformationParameter_LabelType_POINT) {
        CHECK_EQ(label.depth(), typeid(Dtype)==typeid(float)?CV_32F:CV_64F);
        CHECK_EQ(label.channels(), 1);
        CHECK_EQ(label.rows, 1);
        if(has_weight) {
            CHECK_EQ(label.cols%4, 0);
        }
        else {
            CHECK_EQ(label.cols%2, 0);
        }
    }
    else if(label_type==AugTransformationParameter_LabelType_RECT) {
        CHECK_EQ(label.depth(), typeid(Dtype)==typeid(float)?CV_32F:CV_64F);
        CHECK_EQ(label.channels(), 1);
        CHECK_EQ(label.rows, 1);
        if(has_weight) {
            CHECK_EQ(label.cols%10, 0);
            NOT_IMPLEMENTED;
        }
        else {
            CHECK_EQ(label.cols%5, 0);
        }
    }
    else if(label_type==AugTransformationParameter_LabelType_CLASS) {
        CHECK_EQ(label.depth(), typeid(Dtype)==typeid(float)?CV_32F:CV_64F);
        CHECK_EQ(label.channels(), 1);
        CHECK_EQ(label.rows, 1);
        if(has_weight) {
            CHECK_EQ(label.cols%2, 0);
        }
    }
    else if(label_type==AugTransformationParameter_LabelType_SEGMENT) {
        CHECK_EQ(label.rows, input_img_height);
        CHECK_EQ(label.cols, input_img_width);
        if(has_weight) {
            CHECK_EQ(label.rows%2, 0);
            NOT_IMPLEMENTED;
        }
    }
    else if(label_type==AugTransformationParameter_LabelType_AREA) {
        CHECK_EQ(label.depth(), typeid(Dtype)==typeid(float)?CV_32F:CV_64F);
        CHECK_EQ(label.channels(), 1);
        CHECK_EQ(label.rows, 1);
        if(has_weight) {
            CHECK_EQ(label.cols%2, 0);
        }
    }
  
    // get output image size (after affine and crop)     
    int output_img_height;
    int output_img_width;
    if(need_crop) {
        output_img_height=crop_h;
        output_img_width=crop_w;
    }
    else {
        output_img_height=input_img_height;
        output_img_width=input_img_width;
    }

    Mat output_img;
    Mat output_label;
    if(aug_each_channel) {
        vector<Mat> split_input_img(input_img_channels);
        split(img, split_input_img.data());
        vector<Mat> split_output_img(input_img_channels);
        vector<Mat> transform_matrix(input_img_channels);
        for(size_t i=0;i<split_input_img.size();i++) {
            // get augment parameter
            bool aug_flag=aug_rng();
            float trans_x=trans_x_rng();
            float trans_y=trans_y_rng();
            float zoom=zoom_rng();
            float rotate=rotate_rng();
            bool mirror=mirror_rng();

            // affine and crop image and label
            bool need_affine=(zoom!=1.0f || rotate!=0.0f || trans_x!=0.0f || trans_y!=0.0f || mirror) && aug_flag;
            if(need_affine) {
                // calculate affine matrix
                // rotate 
                Point2f center;
                center.x = input_img_width/2.0 - 0.5;
                center.y = input_img_height/2.0 - 0.5;
                transform_matrix[i] = getRotationMatrix2D(center, rotate, 1);
                // translate 
                transform_matrix[i].at<double>(0, 2) += trans_x ;
                transform_matrix[i].at<double>(1, 2) += trans_y ;
                // zoom
                for(int j = 0; j < 3; j++){
                    transform_matrix[i].at<double>(0, j) *= zoom;
                    transform_matrix[i].at<double>(1, j) *= zoom;
                }
                transform_matrix[i].at<double>(0, 2) += (1 - zoom) * center.x;
                transform_matrix[i].at<double>(1, 2) += (1 - zoom) * center.y;

                // if needed, apply crop together with affine to accelerate
                transform_matrix[i].at<double>(0, 2) -= (input_img_width-output_img_width)/2.0;
                transform_matrix[i].at<double>(1, 2) -= (input_img_height-output_img_height)/2.0;

                // mirror about x axis in cropped image
                if (mirror) {
                    transform_matrix[i].at<double>(0, 0) = -transform_matrix[i].at<double>(0, 0);
                    transform_matrix[i].at<double>(0, 1) = -transform_matrix[i].at<double>(0, 1);
                    transform_matrix[i].at<double>(0, 2) = output_img_width-transform_matrix[i].at<double>(0, 2);
                }

                // transform image
                warpAffine(split_input_img[i], split_output_img[i], transform_matrix[i], Size(output_img_width, output_img_height),
                        INTER_LINEAR, BORDER_CONSTANT, Scalar(127));
            }
            else {
                // crop image
                int crop_offset_x=(input_img_width-output_img_width)/2;
                int crop_offset_y=(input_img_height-output_img_height)/2;
                Rect crop_rect=Rect(crop_offset_x, crop_offset_y, output_img_width, output_img_height);
                if((crop_rect&Rect(0, 0, split_input_img[i].cols, split_input_img[i].rows)).area()==crop_rect.area()) {
                    split_output_img[i]=split_input_img[i](crop_rect).clone();
                }
                else {
                    NOT_IMPLEMENTED;
                }
            }

            // random occlusion
            if(aug_flag) {
                int occlusion_width = occlusion_width_rng();
                int occlusion_height  = occlusion_height_rng();
                Rect occlusion_rect(occlusion_center_x_rng()-occlusion_width/2, occlusion_center_y_rng()-occlusion_height/2,
                        occlusion_width, occlusion_height);
                split_output_img[i](occlusion_rect&Rect(0, 0, output_img.cols, output_img.rows)).setTo(Scalar(occlusion_color_rng()));
            }

            // random noise
            if(aug_flag && param_.noise()) {
                int num=split_output_img[i].rows*split_output_img[i].cols*split_output_img[i].channels();
                unsigned char* data=split_output_img[i].data;
                for(int i=0;i<num;i++) {
                    data[i]=std::max(std::min(data[i]+noise_rng(), 255), 0);
                }
            }
        }
        merge(split_output_img, output_img);

        // transform label
        if(label_type==AugTransformationParameter_LabelType_CLASS) {
            output_label=label.clone();
        }
        else {
            NOT_IMPLEMENTED;
        }
    }
    else {
        // get augment parameter
        bool aug_flag=aug_rng();
        float trans_x=trans_x_rng();
        float trans_y=trans_y_rng();
        float zoom=zoom_rng();
        float rotate=rotate_rng();
        bool mirror=mirror_rng();

        // affine and crop image and label
        bool need_affine=(zoom!=1.0f || rotate!=0.0f || trans_x!=0.0f || trans_y!=0.0f || mirror) && aug_flag;
        if(need_affine) {
            // calculate affine matrix
            // rotate
            Point2f center;
            center.x = input_img_width/2.0 - 0.5;
            center.y = input_img_height/2.0 - 0.5;
            Mat transform_matrix =  getRotationMatrix2D(center, rotate, 1);
            // translate
            transform_matrix.at<double>(0, 2) += trans_x ;
            transform_matrix.at<double>(1, 2) += trans_y ;
            // zoom
            for(int i = 0; i < 3; i++){
                transform_matrix.at<double>(0, i) *= zoom;
                transform_matrix.at<double>(1, i) *= zoom;
            }
            transform_matrix.at<double>(0, 2) += (1 - zoom) * center.x;
            transform_matrix.at<double>(1, 2) += (1 - zoom) * center.y;

            // if needed, apply crop together with affine to accelerate
            transform_matrix.at<double>(0, 2) -= (input_img_width-output_img_width)/2.0;
            transform_matrix.at<double>(1, 2) -= (input_img_height-output_img_height)/2.0;

            // mirror about x axis in cropped image
            if (mirror) {
                transform_matrix.at<double>(0, 0) = -transform_matrix.at<double>(0, 0);
                transform_matrix.at<double>(0, 1) = -transform_matrix.at<double>(0, 1);
                transform_matrix.at<double>(0, 2) = output_img_width-transform_matrix.at<double>(0, 2);
            }

            // transform image
            WarpAffineMultiChannels(img, output_img, transform_matrix, Size(output_img_width, output_img_height),
                    INTER_LINEAR, BORDER_CONSTANT, Scalar(127, 127, 127, 0));

            // transform label
            if(label_type==AugTransformationParameter_LabelType_POINT) {
                output_label.create(1, label.cols, typeid(Dtype)==typeid(float)?CV_32F:CV_64F);
                // transform point
                if(has_weight) {
                    for(int i=0;i<label.cols/2;i+=2) {
                        Point2f point(label.at<Dtype>(0, i), label.at<Dtype>(0, i+1));
                        AffinePoint(point, transform_matrix);
                        output_label.at<Dtype>(0, i)=point.x;
                        output_label.at<Dtype>(0, i+1)=point.y;
                    }
                    for(int i=label.cols/2;i<label.cols;i++) {
                        output_label.at<Dtype>(0, i)=label.at<Dtype>(0, i);
                    }
                }
                else {
                    for(int i=0;i<label.cols;i+=2) {
                        Point2f point(label.at<Dtype>(0, i), label.at<Dtype>(0, i+1));
                        AffinePoint(point, transform_matrix);
                        output_label.at<Dtype>(0, i)=point.x;
                        output_label.at<Dtype>(0, i+1)=point.y;
                    }
                }
            }
            else if(label_type==AugTransformationParameter_LabelType_RECT) {
                output_label.create(1, label.cols, typeid(Dtype)==typeid(float)?CV_32F:CV_64F);
                // transform rectangle
                for(int i=0;i<label.cols;i+=5) {
                    output_label.at<Dtype>(0, i)=label.at<Dtype>(0, i);
                    if(label.at<Dtype>(0, i)<0) {
                        continue;
                    }
                    Point2f left_top(label.at<Dtype>(0, i+1), label.at<Dtype>(0, i+2));
                    Point2f right_bottom(label.at<Dtype>(0, i+3), label.at<Dtype>(0, i+4));
                    AffineRect(left_top, right_bottom, transform_matrix);
                    output_label.at<Dtype>(0, i+1)=left_top.x;
                    output_label.at<Dtype>(0, i+2)=left_top.y;
                    output_label.at<Dtype>(0, i+3)=right_bottom.x;
                    output_label.at<Dtype>(0, i+4)=right_bottom.y;
                }
            }
            else if(label_type==AugTransformationParameter_LabelType_CLASS) {
                output_label=label.clone();
            }
            else if(label_type==AugTransformationParameter_LabelType_SEGMENT) {
                WarpAffineMultiChannels(label, output_label, transform_matrix, Size(output_img_width, output_img_height),
                    INTER_NEAREST, BORDER_CONSTANT, Scalar(0));
                vector<Mat> split_output_label(output_label.channels());
                split(output_label, split_output_label.data());
                for(auto& channel: split_output_label) {
                    channel.convertTo(channel, CV_32F);
                }
                merge(split_output_label, output_label);
                int ele_num=output_label.cols*output_label.rows*output_label.channels();
                Dtype* output_label_data=(Dtype*)output_label.data;
                for(int i=0;i<ele_num;i++) {
                    output_label_data[i]=output_label_data[i]>static_cast<Dtype>(0)?static_cast<Dtype>(1):static_cast<Dtype>(0);
                }
            }
            else if(label_type==AugTransformationParameter_LabelType_AREA) {
                output_label.create(1, label.cols, typeid(Dtype)==typeid(float)?CV_32F:CV_64F);
                for(int i=0;i<label.cols;i++) {
                    output_label.at<Dtype>(0, i)=label.at<Dtype>(0, i)*zoom*zoom;
                }
            }
            else {
                NOT_IMPLEMENTED;
            }
        }
        else {
            // crop image
            int crop_offset_x=(input_img_width-output_img_width)/2;
            int crop_offset_y=(input_img_height-output_img_height)/2;
            Rect crop_rect=Rect(crop_offset_x, crop_offset_y, output_img_width, output_img_height);
            if((crop_rect&Rect(0, 0, img.cols, img.rows)).area()==crop_rect.area()) {
                output_img=img(crop_rect).clone();
            }
            else {
                NOT_IMPLEMENTED;
            }

            // crop label
            if(label_type==AugTransformationParameter_LabelType_POINT) {
                output_label.create(1, label.cols, typeid(Dtype)==typeid(float)?CV_32F:CV_64F);
                if(has_weight) {
                    for(int i=0;i<label.cols/2;i+=2) {
                        output_label.at<Dtype>(0, i)=label.at<Dtype>(0, i)-crop_offset_x;
                        output_label.at<Dtype>(0, i+1)=label.at<Dtype>(0, i+1)-crop_offset_y;
                    }
                    for(int i=label.cols/2;i<label.cols;i++) {
                        output_label.at<Dtype>(0, i)=label.at<Dtype>(0, i);
                    }
                }
                else {
                    for(int i=0;i<label.cols;i+=2) {
                        output_label.at<Dtype>(0, i)=label.at<Dtype>(0, i)-crop_offset_x;
                        output_label.at<Dtype>(0, i+1)=label.at<Dtype>(0, i+1)-crop_offset_y;
                    }
                }
            }
            else if(label_type==AugTransformationParameter_LabelType_RECT) {
                output_label.create(1, label.cols, typeid(Dtype)==typeid(float)?CV_32F:CV_64F);
                for(int i=0;i<label.cols;i+=5) {
                    output_label.at<Dtype>(0, i)=label.at<Dtype>(0, i);
                    if(label.at<Dtype>(0, i)<0) {
                        continue;
                    }
                    output_label.at<Dtype>(0, i+1)=label.at<Dtype>(0, i+1)-crop_offset_x;
                    output_label.at<Dtype>(0, i+2)=label.at<Dtype>(0, i+2)-crop_offset_y;
                    output_label.at<Dtype>(0, i+3)=label.at<Dtype>(0, i+3)-crop_offset_x;
                    output_label.at<Dtype>(0, i+4)=label.at<Dtype>(0, i+4)-crop_offset_y;
                }
            }
            else if(label_type==AugTransformationParameter_LabelType_CLASS) {
                output_label=label.clone();
            }
            else if(label_type==AugTransformationParameter_LabelType_SEGMENT) {
                if((crop_rect&Rect(0, 0, label.cols, label.rows)).area()==crop_rect.area()) {
                    output_label=label(crop_rect).clone();
                }
                else {
                    NOT_IMPLEMENTED;
                }
                vector<Mat> split_output_label(output_label.channels());
                split(output_label, split_output_label.data());
                for(auto& channel: split_output_label) {
                    channel.convertTo(channel, CV_32F);
                }
                merge(split_output_label, output_label);
                int ele_num=output_label.cols*output_label.rows*output_label.channels();
                Dtype* output_label_data=(Dtype*)output_label.data;
                for(int i=0;i<ele_num;i++) {
                    output_label_data[i]=output_label_data[i]>static_cast<Dtype>(0)?static_cast<Dtype>(1):static_cast<Dtype>(0);
                }
            }
            else if(label_type==AugTransformationParameter_LabelType_AREA) {
                output_label=label.clone();
            }
            else {
                NOT_IMPLEMENTED;
            }
        }

        // if apply mirror, the label's order may be changed
        if(mirror) {
            // for Point label, the order of the Point list may changed by mirror due to the Point's defination
            // this is specified by corr_list in prototxt.
            if(label_type==AugTransformationParameter_LabelType_POINT) {
                for(size_t i=0;i<corr_list.size();i+=2) {
                    CHECK_LT(corr_list[i], output_label.cols/2);
                    CHECK_LT(corr_list[i+1], output_label.cols/2);
                    Dtype temp;
                    temp=output_label.at<Dtype>(0, corr_list[i]*2);
                    output_label.at<Dtype>(0, corr_list[i]*2)=output_label.at<Dtype>(0, corr_list[i+1]*2);
                    output_label.at<Dtype>(0, corr_list[i+1]*2)=temp;
                    temp=output_label.at<Dtype>(0, corr_list[i]*2+1);
                    output_label.at<Dtype>(0, corr_list[i]*2+1)=output_label.at<Dtype>(0, corr_list[i+1]*2+1);
                    output_label.at<Dtype>(0, corr_list[i+1]*2+1)=temp;
                }
            }
            // for Rect label, the order of left and right border will changed by mirror
            // but the left and right border are figured out after mirror. So we don't need to change them.
            else if(label_type==AugTransformationParameter_LabelType_RECT) {
            }
            else if(label_type==AugTransformationParameter_LabelType_CLASS) {
            }
            else if(label_type==AugTransformationParameter_LabelType_SEGMENT) {
            }
            else if(label_type==AugTransformationParameter_LabelType_AREA) {
            }
            else {
                NOT_IMPLEMENTED;
            }
        }
        
        // random occlusion
        if(aug_flag) {
            int occlusion_width = occlusion_width_rng();
            int occlusion_height  = occlusion_height_rng();
            Rect occlusion_rect(occlusion_center_x_rng()-occlusion_width/2, occlusion_center_y_rng()-occlusion_height/2,
                    occlusion_width, occlusion_height);
            output_img(occlusion_rect&Rect(0, 0, output_img.cols, output_img.rows)).setTo(Scalar(occlusion_color_rng(),
                    occlusion_color_rng(), occlusion_color_rng()));
        }

        // random noise
        if(aug_flag && param_.noise()) {
            int num=output_img.rows*output_img.cols*output_img.channels();
            unsigned char* data=output_img.data;
            for(int i=0;i<num;i++) {
                data[i]=std::max(std::min(data[i]+noise_rng(), 255), 0);
            }
        }
    }

    CHECK(output_img.data);
    CHECK(output_label.data);
  
    // convert to Dtype, and normalize
    vector<Mat> split_img(output_img.channels());
    split(output_img, &(split_img[0]));
    Mat mean=Mat::zeros(output_img.channels(), 1, CV_64F);
    Mat std=Mat::ones(output_img.channels(), 1, CV_64F);
    const bool normalize = param_.normalize();
    if(normalize) {
        meanStdDev(output_img, mean, std);
        for(size_t i=0;i<output_img.channels();i++) {
            if(std.at<double>(i, 0)<1E-6) {
                std.at<double>(i, 0)=1;
            }
        }
    }
    for(size_t i=0;i<input_img_channels;i++) {
        split_img[i].convertTo(split_img[i], typeid(Dtype)==typeid(float)?CV_32F:CV_64F, 
                1.0/std.at<double>(i, 0), -1*mean.at<double>(i, 0)/std.at<double>(i, 0));
    }
  
    // reshape and copy image and label to output blobs
    transformed_img.Reshape(1, output_img.channels(), output_img.rows, output_img.cols);
    Dtype* transformed_img_data = transformed_img.mutable_cpu_data();
    for(size_t i=0;i<output_img.channels();i++) {
        caffe_copy(output_img.rows*output_img.cols, (Dtype*)split_img[i].data, 
                transformed_img_data+i*output_img.rows*output_img.cols);
    }
    if(label_type==AugTransformationParameter_LabelType_SEGMENT) {
        transformed_label.Reshape(1, output_label.channels(), output_label.rows, output_label.cols);
        Dtype* transformed_label_data = transformed_label.mutable_cpu_data();
        vector<Mat> split_label(output_label.channels());
        split(output_label, split_label.data());
        for(size_t i=0;i<output_label.channels();i++) {
            caffe_copy(output_label.rows*output_label.cols, (Dtype*)split_label[i].data, 
                    transformed_label_data+i*output_label.rows*output_label.cols);
        }
    }
    else {
        vector<int> transformed_label_shape(2);
        transformed_label_shape[0]=1;
        transformed_label_shape[1]=output_label.cols;
        transformed_label.Reshape(transformed_label_shape);
        Dtype* transformed_label_data = transformed_label.mutable_cpu_data();
        caffe_copy(output_label.cols, (Dtype*)output_label.data, transformed_label_data);
    }
}

template <typename Dtype>
void AugDataTransformer<Dtype>::InitRand() {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new mt19937(rng_seed));

    aug_rng=bind(bernoulli_distribution(param_.aug_prob()), mt19937((*rng_)()));

    if(param_.zoom()) {
        if(param_.has_zoom_scale()) {
            zoom_rng=bind([](float input){return input;}, param_.zoom_scale());
        }
        else {
            if(param_.zoom_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                zoom_rng=bind(uniform_real_distribution<float>(param_.zoom_rng().min(), param_.zoom_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }
    }
    else {
        zoom_rng=[](){return 1.0f;};
    }

    if(param_.trans()) {
        if(param_.has_trans_x()) {
            trans_x_rng=bind([](float input){return input;}, param_.trans_x());
        }
        else if(param_.has_trans_x_rng()) {
            if(param_.trans_x_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                trans_x_rng=bind(uniform_real_distribution<float>(param_.trans_x_rng().min(), param_.trans_x_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }
        else {
            if(param_.trans_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                trans_x_rng=bind(uniform_real_distribution<float>(param_.trans_rng().min(), param_.trans_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }

        if(param_.has_trans_y()) {
            trans_y_rng=bind([](float input){return input;}, param_.trans_y());
        }
        else if(param_.has_trans_y_rng()) {
            if(param_.trans_y_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                trans_y_rng=bind(uniform_real_distribution<float>(param_.trans_y_rng().min(), param_.trans_y_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }
        else {
            if(param_.trans_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                trans_y_rng=bind(uniform_real_distribution<float>(param_.trans_rng().min(), param_.trans_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }
    }
    else {
        trans_x_rng=[](){return 0.0f;};
        trans_y_rng=[](){return 0.0f;};
    }

    if(param_.rotate()) {
        if(param_.has_rotate_deg()) {
            rotate_rng=bind([](float input){return input;}, param_.rotate_deg());
        }
        else {
            if(param_.rotate_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                rotate_rng=bind(uniform_real_distribution<float>(param_.rotate_rng().min(), param_.rotate_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }
    }
    else {
        rotate_rng=[](){return 0.0f;};
    }

    if(param_.mirror()) {
        mirror_rng=bind(bernoulli_distribution(0.5), *rng_);
    }
    else {
        mirror_rng=[](){return false;};
    }

    if(param_.occlusion()) {
        if(param_.has_occlusion_width()) {
            occlusion_width_rng=bind([](int input){return input;}, param_.occlusion_width());
        }
        else if(param_.has_occlusion_width_rng()) {
            if(param_.occlusion_width_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                occlusion_width_rng=bind(uniform_int_distribution<int>(param_.occlusion_width_rng().min(), param_.occlusion_width_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }
        else {
            if(param_.occlusion_size_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                occlusion_width_rng=bind(uniform_int_distribution<int>(param_.occlusion_size_rng().min(), param_.occlusion_size_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }

        if(param_.has_occlusion_height()) {
            occlusion_height_rng=bind([](int input){return input;}, param_.occlusion_height());
        }
        else if(param_.has_occlusion_height_rng()) {
            if(param_.occlusion_height_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                occlusion_height_rng=bind(uniform_int_distribution<int>(param_.occlusion_height_rng().min(), param_.occlusion_height_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }
        else {
            if(param_.occlusion_size_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                occlusion_height_rng=bind(uniform_int_distribution<int>(param_.occlusion_size_rng().min(), param_.occlusion_size_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }

        if(param_.has_occlusion_center_x()) {
            occlusion_center_x_rng=bind([](int input){return input;}, param_.occlusion_center_x());
        }
        else if(param_.has_occlusion_center_x_rng()) {
            if(param_.occlusion_center_x_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                occlusion_center_x_rng=bind(uniform_int_distribution<int>(param_.occlusion_center_x_rng().min(), param_.occlusion_center_x_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }
        else {
            if(param_.occlusion_center_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                occlusion_center_x_rng=bind(uniform_int_distribution<int>(param_.occlusion_center_rng().min(), param_.occlusion_center_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }

        if(param_.has_occlusion_center_y()) {
            occlusion_center_y_rng=bind([](int input){return input;}, param_.occlusion_center_y());
        }
        else if(param_.has_occlusion_center_y_rng()) {
            if(param_.occlusion_center_y_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                occlusion_center_y_rng=bind(uniform_int_distribution<int>(param_.occlusion_center_y_rng().min(), param_.occlusion_center_y_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }
        else {
            if(param_.occlusion_center_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                occlusion_center_y_rng=bind(uniform_int_distribution<int>(param_.occlusion_center_rng().min(), param_.occlusion_center_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }

        if(param_.has_occlusion_color()) {
            occlusion_color_rng=bind([](int input){return input;}, param_.occlusion_color());
        }
        else {
            if(param_.occlusion_color_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
                occlusion_color_rng=bind(uniform_int_distribution<int>(param_.occlusion_color_rng().min(), param_.occlusion_color_rng().max()), mt19937((*rng_)()));
            }
            else {
                NOT_IMPLEMENTED;
            }
        }
    }
    else {
        occlusion_width_rng=[](){return 0;};
        occlusion_height_rng=[](){return 0;};
        occlusion_center_x_rng=[](){return 0;};
        occlusion_center_y_rng=[](){return 0;};
        occlusion_color_rng=[](){return 0;};
    }

    if(param_.noise()) {
        if(param_.noise_rng().type() == MRNGParameter_MRNGType_UNIFORM) {
            noise_rng=bind(uniform_int_distribution<int>(param_.noise_rng().min(), param_.noise_rng().max()), mt19937((*rng_)()));
        }
        else {
            NOT_IMPLEMENTED;
        }
    }
    else {
        noise_rng=[](){return 0;};
    }
}

INSTANTIATE_CLASS(AugDataTransformer);

}  // namespace caffe
