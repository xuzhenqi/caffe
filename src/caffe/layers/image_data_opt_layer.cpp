#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <boost/random.hpp>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

    template <typename Dtype>
    ImageDataOptLayer<Dtype>::~ImageDataOptLayer<Dtype>() {
        this->StopInternalThread();
    }

    template <typename Dtype>
    void ImageDataOptLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype>*> &bottom,
            const vector<Blob<Dtype>*> &top) {
        BaseDataLayer<Dtype>::LayerSetUp(bottom, top); // will call DataLayerSetUp
        // Now, start the prefetch thread. Before calling prefetch, we make three
        // cpu_data calls so that the prefetch thread does not accidentally make
        // simultaneous cudaMalloc calls when the main thread is running. In some
        // GPUs this seems to cause failures if we do not so.
        this->prefetch_data_[0]->mutable_cpu_data();
        this->prefetch_data_[1]->mutable_cpu_data();

        DLOG(INFO) << "Initializing prefetch";
        this->StartInternalThread();
        DLOG(INFO) << "Prefetch initialized.";
    }

    template <typename Dtype>
    void ImageDataOptLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
        this->output_labels_ = true;

        const int new_height = this->layer_param_.image_data_param().new_height();
        const int new_width  = this->layer_param_.image_data_param().new_width();
        const bool is_color  = this->layer_param_.image_data_param().is_color();
        string root_folder = this->layer_param_.image_data_param().root_folder();

        CHECK((new_height == 0 && new_width == 0) ||
              (new_height > 0 && new_width > 0)) << "Current implementation requires "
                "new_height and new_width to be set at the same time.";
        // Read the file with filenames and labels
        const string& source = this->layer_param_.image_data_param().source();
        LOG(INFO) << "Opening file " << source;
        std::cout << "opening file " << source << std::endl;
        std::ifstream infile(source.c_str());
        string filename;
        int label, frame;
        while (infile >> filename >> label >> frame) {
            lines_.push_back(std::make_pair(filename, label));
            frames_.push_back(frame - 1); // For optical flow, ignore the last frame.
        }

        if (this->layer_param_.image_data_param().shuffle()) {
            LOG(FATAL) << "Shuffle is not supported";
        }
        LOG(INFO) << "A total of " << lines_.size() << " videos.";
        //std::cout << "A total of " << lines_.size() << " videos." << std::endl;

        lines_id_ = 0;
        // Check if we would need to randomly skip a few data points
        if (this->layer_param_.image_data_param().rand_skip()) {
            LOG(FATAL) << "Skip is not supported.";
        }
        // Read an image, and use it to initialize the top blob.
        cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first + "_1_x.png",
                                          new_height, new_width, is_color);
        CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first + "_1_x.png";
        // Use data_transformer to infer the expected blob shape from a cv_image.
        vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
        top_shape[1] *= 2;
        this->transformed_data_.Reshape(top_shape);
        // Reshape prefetch_data and top[0] according to the batch_size.
        const int batch_size = this->layer_param_.image_data_param().batch_size();
        CHECK_GT(batch_size, 0) << "Positive batch size required";
        top_shape[0] = batch_size;
        top[0]->Reshape(top_shape);
        this->prefetch_data_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
        this->prefetch_data_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
        this->prefetch_data_[0]->Reshape(top_shape);
        LOG(INFO) << "output data size: " << top[0]->num() << ","
        << top[0]->channels() << "," << top[0]->height() << ","
        << top[0]->width();
        // label
        vector<int> label_shape(1, batch_size);
        top[1]->Reshape(label_shape);
        this->prefetch_data_[1]->Reshape(label_shape);

        fps_ = this->layer_param_.image_data_rnn_param().fps();
        CHECK(current_frame_.empty());
        current_frame_.insert(current_frame_.begin(), lines_.size(), 1);
        const unsigned int prefetch_rng_seed = caffe_rng_rand();
        prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
        caffe::rng_t* prefetch_rng =
                static_cast<caffe::rng_t*>(prefetch_rng_->generator());
        unifor_gen.reset(new boost::uniform_int<int>(0, lines_.size() - 1));
    }

    template <typename Dtype>
    void ImageDataOptLayer<Dtype>::InternalThreadEntry() {
        CPUTimer batch_timer;
        batch_timer.Start();
        double trans_time = 0;
        CPUTimer timer;
        CHECK(this->prefetch_data_[0]->count());
        CHECK(this->prefetch_data_[1]->count());
        CHECK(this->transformed_data_.count());
        ImageDataParameter image_data_param = this->layer_param_.image_data_param();
        const int batch_size = image_data_param.batch_size();
        const int new_height = image_data_param.new_height();
        const int new_width = image_data_param.new_width();
        const bool is_color = image_data_param.is_color();
        string root_folder = image_data_param.root_folder();
        string filename1, filename2;

        caffe::rng_t* prefetch_rng =
                static_cast<caffe::rng_t*>(prefetch_rng_->generator());
        Dtype* prefetch_label = this->prefetch_data_[1]->mutable_cpu_data();

        int line_id_temp, offset;
        Datum datum;
        vector<string> filenames(2, "");
        for (int i = 0; i < batch_size; ++i) {
            line_id_temp = (*unifor_gen)(*prefetch_rng);
            // get a blob
            timer.Start();
            filenames[0] = root_folder + lines_[line_id_temp].first + "_" +
                        boost::lexical_cast<string>(current_frame_[line_id_temp]) + "_x.png";
            filenames[1] = root_folder + lines_[line_id_temp].first + "_" +
                        boost::lexical_cast<string>(current_frame_[line_id_temp]) + "_y.png";

            ReadImagesToDatum(filenames, &datum, new_height, new_width, is_color);
            // Apply transformations (mirror, crop...) to the image
            offset = this->prefetch_data_[0]->offset(i);
            this->transformed_data_.set_cpu_data(
                    prefetch_data_[0]->mutable_cpu_data() + offset);
            this->data_transformer_->Transform(datum, &(this->transformed_data_));
            // Reading label
            prefetch_label[i] = lines_[line_id_temp].second;
            current_frame_[line_id_temp] += fps_;
            if (current_frame_[line_id_temp] > frames_[line_id_temp]) {
                current_frame_[line_id_temp] = 1;
            }
            trans_time += timer.MicroSeconds();
        }
        //std::cout << filenames[0] << "\t\t" << filenames[1] << "\t\t" << prefetch_label[4] << "\t\t"
        //    << current_frame_[4] << "\t\t" << std::endl;
        batch_timer.Stop();
        DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }

    template <typename Dtype>
    void ImageDataOptLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        // First, join the thread
        this->StopInternalThread();
        DLOG(INFO) << "Thread joined";
        // Reshape to loaded data.
        for (int i = 0; i < top.size(); ++i) {
            top[i]->Reshape(this->prefetch_data_[i]->num(),
                            this->prefetch_data_[i]->channels(),
                            this->prefetch_data_[i]->height(),
                            this->prefetch_data_[i]->width());
            // Copy the data
            caffe_copy(prefetch_data_[i]->count(), prefetch_data_[i]->cpu_data(),
                       top[i]->mutable_cpu_data());
            DLOG(INFO) << "Prefetch copied";
        }
        // Start a new prefetch thread
        DLOG(INFO) << "StartInternalThread";
        this->StartInternalThread();
    }

#ifdef CPU_ONLY
    STUB_GPU_FORWARD(TripletImageDataLayer, Forward);
#endif

    INSTANTIATE_CLASS(ImageDataOptLayer);
    REGISTER_LAYER_CLASS(ImageDataOpt);

}  // namespace caffe
