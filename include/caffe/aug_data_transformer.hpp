#ifndef CAFFE_AUG_DATA_TRANSFORMER_HPP
#define CAFFE_AUG_DATA_TRANSFORMER_HPP

#include <vector>
#include <random>
#include <functional>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class AugDataTransformer {
public:
    explicit AugDataTransformer(const AugTransformationParameter& param, Phase phase);
    virtual ~AugDataTransformer() {}

    void InitRand();
    void Transform(const cv::Mat& img, const cv::Mat& label, Blob<Dtype>& transformed_img, Blob<Dtype>& transformed_label);

protected:
    static void AffinePoint(cv::Point2f& point, const cv::Mat& affine_mat);
    static void AffineRect(cv::Point2f& left_top, cv::Point2f& right_bottom, const cv::Mat& affine_mat);
    static void WarpAffineMultiChannels(const cv::Mat& input, cv::Mat& output,
            const cv::Mat& trans_mat, const cv::Size& output_size, int interp_type=cv::INTER_LINEAR, int border_type=cv::BORDER_CONSTANT,
            const cv::Scalar& border_value=cv::Scalar::all(0));

    AugTransformationParameter param_;
    shared_ptr<std::mt19937> rng_;
    Phase phase_;
    AugTransformationParameter_LabelType label_type;

    bool need_crop;
    int crop_w;
    int crop_h;

    bool has_weight;
    bool aug_each_channel;

    std::function<bool()> aug_rng;
    std::function<float()> trans_x_rng;
    std::function<float()> trans_y_rng;
    std::function<float()> zoom_rng;
    std::function<float()> rotate_rng;
    std::function<bool()> mirror_rng;
    std::function<int()> occlusion_width_rng;
    std::function<int()> occlusion_height_rng;
    std::function<int()> occlusion_center_x_rng;
    std::function<int()> occlusion_center_y_rng;
    std::function<int()> occlusion_color_rng;
    std::function<int()> noise_rng;

    std::vector<size_t> corr_list;
};

}
#endif

