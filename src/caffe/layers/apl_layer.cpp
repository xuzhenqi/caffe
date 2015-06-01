// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/io.hpp"
#include <ctime>
#include <cstdio>
#include <string.h>
#include <locale>

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
	void APLLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		CHECK_GE(bottom[0]->num_axes(), 2)
			<< "Number of axes of bottom blob must be >=2.";

		// Figure out the dimensions
		M_ = bottom[0]->num();
		K_ = bottom[0]->count() / bottom[0]->num();
		N_ = K_;

		sums_ = this->layer_param_.apl_param().sums();
    slope_num = sums_;
		save_mem_ = this->layer_param_.apl_param().save_mem();

    if (this->layer_param_.apl_param().has_slope_sum_constrains()) {
      --slope_num;
      slope_last.Reshape(1, 1, 1, K_);
    }

		// Check if we need to set up the weights
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		} 

		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		} else {
			this->blobs_.resize(2);

			shared_ptr<Filler<Dtype> > slope_filler;
			if (this->layer_param_.apl_param().has_slope_filler()) {
				slope_filler.reset(GetFiller<Dtype>(this->layer_param_.apl_param().slope_filler()));
			} else {
				FillerParameter slope_filler_param;
				slope_filler_param.set_type("uniform");
				slope_filler_param.set_min((Dtype) -0.5/((Dtype) sums_));
				slope_filler_param.set_max((Dtype)  0.5/((Dtype) sums_));
				slope_filler.reset(GetFiller<Dtype>(slope_filler_param));
			}
			//shared_ptr<Filler<Dtype> > slope_filler(GetFiller<Dtype>(
			//		this->layer_param_.apl_param().slope_filler()));
			shared_ptr<Filler<Dtype> > offset_filler;
			if (this->layer_param_.apl_param().has_offset_filler()) {
				offset_filler.reset(GetFiller<Dtype>(this->layer_param_.apl_param().offset_filler()));
			} else {
				FillerParameter offset_filler_param;
				offset_filler_param.set_type("gaussian");
				offset_filler_param.set_std(0.5);
				offset_filler.reset(GetFiller<Dtype>(offset_filler_param));
			}
			//shared_ptr<Filler<Dtype> > offset_filler(GetFiller<Dtype>(
			//		this->layer_param_.apl_param().offset_filler()));

			//slope
			this->blobs_[0].reset(new Blob<Dtype>(1, 1, slope_num, K_));
			CHECK(this->blobs_[0].get()->count());
			slope_filler->Fill(this->blobs_[0].get());

			//offset
			this->blobs_[1].reset(new Blob<Dtype>(1, 1, sums_, K_));
			CHECK(this->blobs_[1].get()->count());
			offset_filler->Fill(this->blobs_[1].get());
		}


		if (!save_mem_) {
			temp_ex_neuron_sum_.reset(new SyncedMemory(M_ * K_ * sums_ * sizeof(Dtype)));

			example_multiplier_.reset(new SyncedMemory(M_ * sizeof(Dtype)));
			Dtype* example_multiplier_data =
				reinterpret_cast<Dtype*>(example_multiplier_->mutable_cpu_data());
			for (int i = 0; i < M_; ++i) {
				example_multiplier_data[i] = 1.;
			}
		}

		maxs_.reset(new SyncedMemory(M_ * K_ * sums_ * sizeof(Dtype)));

		LOG(INFO) << " Sums: " << sums_;
	}

template <typename Dtype>
	void APLLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		CHECK_GE(bottom[0]->num_axes(), 2)
			<< "Number of axes of bottom blob must be >=2.";
		top[0]->ReshapeLike(*bottom[0]);

		inPlace_memory_.ReshapeLike(*bottom[1]);

    if(this->layer_param_.apl_param().layer_weight_decay() > 1e-9){
      temp1_.ReshapeLike(*(this->blobs_[1]));
      temp2_.ReshapeLike(*(this->blobs_[1]));
      ones_.Reshape(1, 1, sums_, K_);
      caffe_set(ones_.count(), Dtype(1), ones_.mutable_cpu_data());
    }
	}

template <typename Dtype>
void APLLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top){
    if (this->layer_param_.apl_param().has_slope_sum_constrains()) {
      Forward_cpu_v2(bottom, top);
    } else {
      this->Forward_cpu_v1(bottom, top);
    }
}

template <typename Dtype>
	void APLLayer<Dtype>::Forward_cpu_v1(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {

		/* Initialize */
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		const Dtype* neuron_weight = this->blobs_[0]->cpu_data();
		const Dtype* neuron_offset = this->blobs_[1]->cpu_data();
		const int count = bottom[0]->count();

		Dtype* maxs_data = reinterpret_cast<Dtype*>(maxs_->mutable_cpu_data());

		// For in-place computation
		if (bottom[0] == top[0]) {
			caffe_copy(count, bottom_data, inPlace_memory_.mutable_cpu_data());
		}

		/* Forward Prop */
		for (int e=0; e<M_; ++e) {
			int exPos = e*K_;
			int exPosSums = e*K_*sums_;
			for (int k=0; k<K_; ++k) {
				Dtype bottom_data_ex = bottom_data[exPos + k];
				top_data[exPos + k] = max(bottom_data_ex,Dtype(0));

				int sumPos = k*sums_;
				for (int s=0; s<sums_; ++s) {
					maxs_data[exPosSums + sumPos + s] = max(-bottom_data_ex + neuron_offset[sumPos + s], Dtype(0));
					top_data[exPos + k] += neuron_weight[sumPos + s]*maxs_data[exPosSums + sumPos + s];
				}
			}
		}
	}

template <typename Dtype>
void APLLayer<Dtype>::Forward_cpu_v2(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
	/* Initialize */
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	const Dtype* neuron_weight = this->blobs_[0]->cpu_data();
	const Dtype* neuron_offset = this->blobs_[1]->cpu_data();
	const int count = bottom[0]->count();
  
  // Compute the last slope 
  caffe_set(slope_last.count(), Dtype(0.), slope_last.mutable_cpu_data());
  for (int i=0; i<slope_num; ++i){
    caffe_sub(slope_last.count(), slope_last.cpu_data(), neuron_weight+i*K_, 
              slope_last.mutable_cpu_data());
  }
  
  caffe_copy(count, bottom_data, inPlace_memory_.mutable_cpu_data());
  
  caffe_cpu_max(count, inPlace_memory_.cpu_data(), Dtype(0.), top_data);

  for(int i=0; i<sums_; ++i){
    for(int j=0; j<M_; ++j){
      caffe_sub(K_, neuron_weight+i*K_, inPlace_memory_.cpu_data()+j*K_,
                    temp1_.mutable_cpu_data() + j*K_);
      if (i == slope_num){
        caffe_mul(K_, temp1_.cpu_data()+j*K_, slope_last.cpu_data(),
                      temp1_.mutable_cpu_data());
      } else {
        caffe_mul(K_, temp1_.cpu_data()+j*K_, neuron_weight+j*K_,
                      temp1_.mutable_cpu_data());
      }
    }
    caffe_cpu_max(count, temp1_.cpu_data(), Dtype(0), 
                  temp1_.mutable_cpu_data());
    caffe_add(count, top_data, temp1_.cpu_data(), top_data);
  }
}

template <typename Dtype>
void APLLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
  		const vector<Blob<Dtype>*>& bottom) {
  if (this->layer_param_.apl_param().has_slope_sum_constrains()) {
    Backward_cpu_v2(bottom, propagate_down, top);
  } else {
    Backward_cpu_v1(bottom, propagate_down, top);
  }
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	const Dtype* neuron_weight = this->blobs_[0]->cpu_data();
  const Dtype* neuron_offset = this->blobs_[1]->cpu_data();

	Dtype* neuron_weight_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* neuron_offset_diff = this->blobs_[1]->mutable_cpu_diff();
  // Regularise by variance of the weight across different APL units
  if(this->layer_param_.apl_param().layer_weight_decay() > 1e-9){
    string type = this->layer_param_.apl_param().slope_regular_type();
    if (type == string("Variance")){
      Variance_regularise(neuron_weight);
    } else if (type == string("L2")){
      L2_regularise(neuron_weight);
    } else {
      LOG(ERROR) << "Unsupported type: " << type 
          << ". Should be one of [\"Variance\", \"L2\"]";
    }
    caffe_add(temp1_.count(), neuron_weight_diff, temp1_.cpu_data(),
              neuron_weight_diff);
    Variance_regularise(neuron_offset);
    caffe_add(temp1_.count(), neuron_offset_diff, temp1_.cpu_data(),
              neuron_offset_diff);
  } 
}
template <typename Dtype>
	void APLLayer<Dtype>::Backward_cpu_v1(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {
		/* Initialize */
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		const Dtype* neuron_weight = this->blobs_[0]->cpu_data();
    const Dtype* neuron_offset = this->blobs_[1]->cpu_data();

		Dtype* neuron_weight_diff = this->blobs_[0]->mutable_cpu_diff();
		Dtype* neuron_offset_diff = this->blobs_[1]->mutable_cpu_diff();

		const Dtype* maxs_data = reinterpret_cast<const Dtype*>(maxs_->cpu_data());

		// For in-place computation
		if (top[0] == bottom[0]) {
			bottom_data = inPlace_memory_.cpu_data();
		}

		for (int i=0; i < sums_*K_; ++i) {
			neuron_weight_diff[i] = 0;
			neuron_offset_diff[i] = 0;
		}

		/* Gradients to neuron layer*/
		for (int e=0; e<M_; ++e) {
			int exPos = e*K_;
			int exPosSums = e*K_*sums_;

			for (int k=0; k<K_; ++k) {
				Dtype sumTopDiff = top_diff[exPos + k];
				Dtype sumBottomData = bottom_data[exPos + k];			

				//bottom_diff[exPos + k] = sumTopDiff*(sumBottomData > 0);
				bottom_diff[exPos + k] = sumBottomData > 0 ? sumTopDiff : 0;

				int sumPos = k*sums_;
				for (int s=0; s<sums_; ++s) {
					Dtype maxGT_Zero = maxs_data[exPosSums + sumPos + s] > 0;

					Dtype weight_diff = sumTopDiff*maxs_data[exPosSums + sumPos + s];
					Dtype offset_diff = sumTopDiff*neuron_weight[sumPos + s]*maxGT_Zero;

					neuron_weight_diff[sumPos + s] += weight_diff;
					neuron_offset_diff[sumPos + s] += offset_diff;

					//Propagate down gradients to lower layer
					bottom_diff[exPos + k] += -offset_diff;
				}
			}
		}
	
  }
template <typename Dtype>
void APLLayer<Dtype>::Backward_cpu_v2(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {
	/* Initialize */
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	const Dtype* neuron_weight = this->blobs_[0]->cpu_data();
  const Dtype* neuron_offset = this->blobs_[1]->cpu_data();

	Dtype* neuron_weight_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* neuron_offset_diff = this->blobs_[1]->mutable_cpu_diff();
  
  // Computing neuron_weight_diff
  caffe_set(slope_num*K_, Dtype(0.), neuron_weight_diff);

  for (int i=0; i<M_; ++i){
    caffe_sub(K_, neuron_offset+slope_num*K_, bottom_data+i*K_, 
                  temp1_.mutable_cpu_data());
    caffe_cpu_max(K_, temp1_.cpu_data(), Dtype(0.), temp1_.mutable_cpu_data());
    for (int j=0; j<slope_num; ++j){
      caffe_sub(K_, neuron_offset+j*K_, bottom_data+i*K_, 
                    temp2_.mutable_cpu_data()+j*K_);
      caffe_cpu_max(K_, temp2_.cpu_data()+j*K_, Dtype(0.),
                    temp2_.mutable_cpu_data()+j*K_);
      caffe_sub(K_, temp2_.cpu_data()+j*K_, temp1_.cpu_data(), 
                    temp2_.mutable_cpu_data()+j*K_);
      caffe_mul(K_, top_diff+i*K_, temp2_.cpu_data()+j*K_,
                    temp2_.mutable_cpu_data()+j*K_);

    }
    caffe_add(slope_num*K_, temp2_.cpu_data(), neuron_weight_diff, 
                  neuron_weight_diff);
  }
  
  // Computing neuron_offset_diff
  caffe_set(sums_*K_, Dtype(0.), neuron_offset_diff);
  for (int i=0; i<M_; ++i){
    for (int j=0; j<sums_; ++j){
      caffe_cpu_sign(K_, bottom_data+i*K_, temp1_.mutable_cpu_data()+j*K_);
      if (j==slope_num){
        caffe_mul(K_, slope_last.cpu_data(), temp1_.cpu_data()+j*K_,
                      temp1_.mutable_cpu_data()+j*K_);
      } else {
        caffe_mul(K_, neuron_weight+j*K_, temp1_.cpu_data()+j*K_,
                  temp1_.mutable_cpu_data()+j*K_);
      }
      caffe_mul(K_, top_diff+i*K_, temp1_.cpu_data()+j*K_,
                temp1_.mutable_cpu_data()+j*K_);
    }
    caffe_add(sums_*K_, temp1_.cpu_data(), neuron_offset_diff, 
              neuron_offset_diff);
  }

  // Computing bottom_diff
  for (int i=0; i<M_; ++i){
    caffe_cpu_max(K_, bottom_data+i*K_, Dtype(0.), temp1_.mutable_cpu_data());
    for(int j=0; j<sums_; ++j){
      caffe_sub(K_, neuron_offset+j*K_, bottom_data+i*K_, 
                    temp2_.mutable_cpu_data());
      caffe_cpu_max(K_, temp2_.cpu_data(), Dtype(0.), 
                    temp2_.mutable_cpu_data());
      if (j == slope_num) {
        caffe_mul(K_, temp2_.cpu_data(), slope_last.cpu_data(), 
                      temp2_.mutable_cpu_data());
      } else {
        caffe_mul(K_, temp2_.cpu_data(), neuron_weight+j*K_,
                      temp2_.mutable_cpu_data());
      }
      caffe_sub(K_, temp1_.cpu_data(), temp2_.cpu_data(), 
                    temp1_.mutable_cpu_data());
    }
    caffe_mul(K_, top_diff+i*K_, temp1_.cpu_data(), bottom_diff+i*K_);
  }

}
/**
 * Using temp1_ and temp2_ as buffer
 * The outcome will be stored in temp1_.cpu_data()
 */
template <class Dtype>
void APLLayer<Dtype>::Variance_regularise(const Dtype* weight){
      caffe_cpu_gemv<Dtype>(CblasNoTrans, sums_, K_, 1./K_, weight, 
        ones_.cpu_data(), 0., temp1_.mutable_cpu_data()); // Ex
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, sums_, K_, 1, 
          Dtype(1.), temp1_.cpu_data(), ones_.cpu_data(), Dtype(0),
          temp2_.mutable_cpu_data()); 
      caffe_sub(temp2_.count(), weight, temp2_.cpu_data(), 
                temp1_.mutable_cpu_data()); // X - Ex
      caffe_powx(temp1_.count(), weight, Dtype(2.), 
                 temp2_.mutable_cpu_data()); // x^2
      caffe_sub(temp1_.count(), ones_.cpu_data(), temp2_.cpu_data(), 
                temp2_.mutable_cpu_data()); // 1 - x^2
      caffe_mul(temp1_.count(), temp1_.cpu_data(), temp2_.cpu_data(), 
                temp2_.mutable_cpu_data()); // (1 - x^2)(X - Ex)
      caffe_powx(temp1_.count(), temp1_.cpu_data(), Dtype(2.),
                 temp1_.mutable_cpu_data()); // (x - Ex) ^ 2
      caffe_mul(temp1_.count(), temp1_.cpu_data(), weight,
                temp1_.mutable_cpu_data()); // x * (x - Ex)
      caffe_cpu_gemv<Dtype>(CblasNoTrans, sums_, K_, 1./K_, 
          temp2_.cpu_data(), ones_.cpu_data(), 0., 
          temp2_.mutable_cpu_diff()); // E((1-x^2)(x-Ex))
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, sums_, K_,
          1, Dtype(1.), temp2_.cpu_diff(), ones_.cpu_data(), Dtype(0),
          temp1_.mutable_cpu_diff());
      caffe_add(temp1_.count(), temp1_.cpu_diff(), temp1_.cpu_data(),
                temp1_.mutable_cpu_data());
      caffe_sub(temp1_.count(), temp1_.cpu_data(), temp2_.cpu_data(), temp1_.mutable_cpu_data());
      Dtype weight_decay = this->layer_param_.apl_param().layer_weight_decay();
      caffe_scal(temp1_.count(), weight_decay, temp1_.mutable_cpu_data());
}

/**
 * Using temp1_ as buffer
 * The outcome will be stored in temp1_.cpu_data()
 */
template <class Dtype>
void APLLayer<Dtype>::L2_regularise(const Dtype* weight){
  Dtype weight_decay = this->layer_param_.apl_param().layer_weight_decay();
  caffe_cpu_axpby(temp1_.count(), weight_decay, weight, Dtype(0.), temp1_.mutable_cpu_data());
}

#ifdef CPU_ONLY
STUB_GPU(APLLayer);
#endif

INSTANTIATE_CLASS(APLLayer);
REGISTER_LAYER_CLASS(APL);

}  // namespace caffe
