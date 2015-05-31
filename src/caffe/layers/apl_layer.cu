// Copyright 2014 BVLC and contributors.

#include <cublas_v2.h>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
	__global__ void APLForwardSum(const int n, int s, const Dtype* in, Dtype* out, const Dtype* neuron_weight, const Dtype* neuron_offset, Dtype* maxs_data, int sums_, int K_) {
		CUDA_KERNEL_LOOP(index, n) {
			int exPos = ((int) index / K_) * K_;
			int exPosSums = exPos*sums_;
			int k = index % K_;
			int sumPos = k*sums_;

			if (s == 0) {
				out[index] = in[index] > 0 ? in[index] : 0;
			}
			maxs_data[exPosSums + sumPos + s] = max(-in[index] + neuron_offset[sumPos + s], Dtype(0));
			out[index] += neuron_weight[sumPos + s]*maxs_data[exPosSums + sumPos + s];
		}
	}

template <typename Dtype>
	__global__ void APLForwardSumHardcode(const int n, const Dtype* in, Dtype* out, const Dtype* neuron_weight, const Dtype* neuron_offset, Dtype* maxs_data, int sums_, int K_) {
		CUDA_KERNEL_LOOP(index, n) {
			int exPos = ((int) index / K_) * K_;
			int exPosSums = exPos*sums_;
			int k = index % K_;
			int sumPos = k*sums_;

			switch (sums_) {
				case 1 : { 
									 maxs_data[exPosSums + sumPos + 0] = max(-in[index] + neuron_offset[sumPos + 0], Dtype(0));

									 Dtype inMax = in[index] > 0 ? in[index] : 0;
									 out[index] = inMax +  neuron_weight[sumPos + 0]*maxs_data[exPosSums + sumPos + 0];
									 break;
								 }
				case 2 : {
									 maxs_data[exPosSums + sumPos + 0] = max(-in[index] + neuron_offset[sumPos + 0], Dtype(0));
									 maxs_data[exPosSums + sumPos + 1] = max(-in[index] + neuron_offset[sumPos + 1], Dtype(0));

									 Dtype inMax = in[index] > 0 ? in[index] : 0;
									 out[index] = inMax +  neuron_weight[sumPos + 0]*maxs_data[exPosSums + sumPos + 0] + neuron_weight[sumPos + 1]*maxs_data[exPosSums + sumPos + 1];
									 break;
								 }
				case 3 : {
									 maxs_data[exPosSums + sumPos + 0] = max(-in[index] + neuron_offset[sumPos + 0], Dtype(0));
									 maxs_data[exPosSums + sumPos + 1] = max(-in[index] + neuron_offset[sumPos + 1], Dtype(0));
									 maxs_data[exPosSums + sumPos + 2] = max(-in[index] + neuron_offset[sumPos + 2], Dtype(0));

									 Dtype inMax = in[index] > 0 ? in[index] : 0;
									 out[index] = inMax +  neuron_weight[sumPos + 0]*maxs_data[exPosSums + sumPos + 0] + neuron_weight[sumPos + 1]*maxs_data[exPosSums + sumPos + 1] + neuron_weight[sumPos + 2]*maxs_data[exPosSums + sumPos + 2];
									 break;
								 }
				case 4 : {
									 maxs_data[exPosSums + sumPos + 0] = max(-in[index] + neuron_offset[sumPos + 0], Dtype(0));
									 maxs_data[exPosSums + sumPos + 1] = max(-in[index] + neuron_offset[sumPos + 1], Dtype(0));
									 maxs_data[exPosSums + sumPos + 2] = max(-in[index] + neuron_offset[sumPos + 2], Dtype(0));
									 maxs_data[exPosSums + sumPos + 3] = max(-in[index] + neuron_offset[sumPos + 3], Dtype(0));

									 Dtype inMax = in[index] > 0 ? in[index] : 0;
									 out[index] = inMax +  neuron_weight[sumPos + 0]*maxs_data[exPosSums + sumPos + 0] + neuron_weight[sumPos + 1]*maxs_data[exPosSums + sumPos + 1] + neuron_weight[sumPos + 2]*maxs_data[exPosSums + sumPos + 2] + neuron_weight[sumPos + 3]*maxs_data[exPosSums + sumPos + 3];
									 break;
								 }
				case 5 : {
									 maxs_data[exPosSums + sumPos + 0] = max(-in[index] + neuron_offset[sumPos + 0], Dtype(0));
									 maxs_data[exPosSums + sumPos + 1] = max(-in[index] + neuron_offset[sumPos + 1], Dtype(0));
									 maxs_data[exPosSums + sumPos + 2] = max(-in[index] + neuron_offset[sumPos + 2], Dtype(0));
									 maxs_data[exPosSums + sumPos + 3] = max(-in[index] + neuron_offset[sumPos + 3], Dtype(0));
									 maxs_data[exPosSums + sumPos + 4] = max(-in[index] + neuron_offset[sumPos + 4], Dtype(0));

									 Dtype inMax = in[index] > 0 ? in[index] : 0;
									 out[index] = inMax +  neuron_weight[sumPos + 0]*maxs_data[exPosSums + sumPos + 0] + neuron_weight[sumPos + 1]*maxs_data[exPosSums + sumPos + 1] + neuron_weight[sumPos + 2]*maxs_data[exPosSums + sumPos + 2] + neuron_weight[sumPos + 3]*maxs_data[exPosSums + sumPos + 3] + neuron_weight[sumPos + 4]*maxs_data[exPosSums + sumPos + 4];
									 break;
								 }
			}
		}
	}

template <typename Dtype>
	void APLLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		//Forward_cpu(bottom,top);

		//Initialize
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();

		Dtype* maxs_data = reinterpret_cast<Dtype*>(maxs_->mutable_gpu_data());
		const int count = bottom[0]->count();

		//For in-place computation
		if (bottom[0] == top[0]) {
			caffe_copy(count, bottom_data, inPlace_memory_.mutable_gpu_data());
			bottom_data = inPlace_memory_.gpu_data();
		}

		const Dtype* neuron_weight = this->blobs_[0]->gpu_data();
		const Dtype* neuron_offset = this->blobs_[1]->gpu_data();

		for (int s=0; s<sums_; s++) {
			APLForwardSum<Dtype><<<CAFFE_GET_BLOCKS(M_*K_), CAFFE_CUDA_NUM_THREADS>>>(
					M_*K_, s, bottom_data, top_data,neuron_weight,neuron_offset,maxs_data,sums_,K_);
			CUDA_POST_KERNEL_CHECK;
		}
	}

template <typename Dtype>
	__global__ void ComputeDiffExample(int n, int e, Dtype* neuron_weight_diff, Dtype* neuron_offset_diff, const Dtype* neuron_weight, const Dtype* top_diff, const Dtype* bottom_data, const Dtype* maxs_data, int sums_, int K_) {
		CUDA_KERNEL_LOOP(index, n) {
			int k = index / sums_;
			Dtype maxGT_Zero = maxs_data[index] > 0;
			Dtype offset_diff = top_diff[k]*neuron_weight[index]*maxGT_Zero;

			if (e == 0) {
				neuron_weight_diff[index] = top_diff[k]*maxs_data[index];
				neuron_offset_diff[index] = offset_diff;
			} else {
				neuron_weight_diff[index] += top_diff[k]*maxs_data[index];
				neuron_offset_diff[index] += offset_diff;
			}
		}
	}

template <typename Dtype>
	__global__ void ComputeOffsetDiff(int n, Dtype* offset_diff, const Dtype* neuron_weight, const Dtype* top_diff, const Dtype* maxs_data, int sums_, int M_, int K_, int K_Times_Sums) {
		CUDA_KERNEL_LOOP(index, n) {
			Dtype maxGT_Zero = maxs_data[index] > 0;
			offset_diff[index] = top_diff[index / sums_]*neuron_weight[index % K_Times_Sums]*maxGT_Zero;
		}
	}

template <typename Dtype>
	__global__ void ComputeBottomDiffSum(int n, int s, Dtype* bottom_diff, const Dtype* bottom_data, const Dtype* neuron_weight, const Dtype* maxs_data, const Dtype* top_diff, int sums_, int K_) {
		CUDA_KERNEL_LOOP(index, n) {
			int exPos = ((int) index / K_) * K_;
			int exPosSums = exPos*sums_;
			int k = index % K_;
			int sumPos = k*sums_;

			Dtype maxGT_Zero = maxs_data[exPosSums + sumPos + s] > 0;

			Dtype offset_diff = top_diff[index]*neuron_weight[sumPos + s]*maxGT_Zero;

			if (s == 0) {
				bottom_diff[index] = bottom_data[index] > 0 ? top_diff[index] : 0;
			}
			bottom_diff[index] += -offset_diff;
		}
	}

template <typename Dtype>
	__global__ void PropDownMax(int n, Dtype* bottom_diff, const Dtype* bottom_data, const Dtype* top_diff) {
		CUDA_KERNEL_LOOP(index, n) {
			bottom_diff[index] = bottom_data[index] > 0 ? top_diff[index] : 0;
		}
	}

//Mimics Matlab's bsxfun with only mult implemented for now
template <typename Dtype>
	__global__ void bsxfun(int n, const Dtype* mat1, const Dtype* vec, const int mat1dim, const int vectorDim, Dtype* resultMat) {
		CUDA_KERNEL_LOOP(index, n) {
			int vecPos = index % vectorDim;
			int mat1pos2 = index/vectorDim;
			int matLoc = vecPos*mat1dim + mat1pos2;

			resultMat[matLoc] = mat1[matLoc]*vec[vecPos];
		}
	}

template <typename Dtype>
	void APLLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {
		//Backward_cpu(top,propagate_down,bottom);

		//Initialize
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		const int count = bottom[0]->count();
		if (top[0] == bottom[0]) {
			bottom_data = inPlace_memory_.gpu_data();
			caffe_copy(count, top_diff, inPlace_memory_.mutable_gpu_diff());
			top_diff = inPlace_memory_.gpu_diff();
		}

		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		// For in-place computation
		Dtype* maxs_data = reinterpret_cast<Dtype*>(maxs_->mutable_gpu_data());

		//Backprop apl weights
		const Dtype* neuron_weight = this->blobs_[0]->gpu_data();
    const Dtype* neuron_offset = this->blobs_[1]->gpu_data();

		Dtype* neuron_weight_diff = this->blobs_[0]->mutable_gpu_diff();
		Dtype* neuron_offset_diff = this->blobs_[1]->mutable_gpu_diff();

		if (save_mem_) {
			for (int e=0; e<M_; ++e) {
				ComputeDiffExample<Dtype><<<CAFFE_GET_BLOCKS(K_*sums_), CAFFE_CUDA_NUM_THREADS>>>(K_*sums_, e, neuron_weight_diff, neuron_offset_diff, neuron_weight, top_diff + e*K_, bottom_data + e*K_, maxs_data + e*K_*sums_, sums_, K_);
				CUDA_POST_KERNEL_CHECK;
			}
		} else {

			Dtype* temp_ex_neuron_sum = reinterpret_cast<Dtype*>(temp_ex_neuron_sum_->mutable_gpu_data());
			const Dtype* example_multiplier = reinterpret_cast<const Dtype*>(example_multiplier_->gpu_data());

			//Compute derivative for neuron_weight
			bsxfun<Dtype><<<CAFFE_GET_BLOCKS(M_*K_*sums_), CAFFE_CUDA_NUM_THREADS>>>(M_*K_*sums_, maxs_data, top_diff, sums_, M_*K_, temp_ex_neuron_sum);
			CUDA_POST_KERNEL_CHECK;

			caffe_gpu_gemv<Dtype>(CblasTrans, M_, K_*sums_, (Dtype)1., temp_ex_neuron_sum,
					example_multiplier, (Dtype)0., neuron_weight_diff);

			//Compute derivative for neuron_offset
			ComputeOffsetDiff<Dtype><<<CAFFE_GET_BLOCKS(M_*K_*sums_), CAFFE_CUDA_NUM_THREADS>>>(M_*K_*sums_, temp_ex_neuron_sum, neuron_weight, top_diff, maxs_data, sums_, M_, K_, K_*sums_);
			CUDA_POST_KERNEL_CHECK;

			caffe_gpu_gemv<Dtype>(CblasTrans, M_, K_*sums_, (Dtype)1., temp_ex_neuron_sum,
					example_multiplier, (Dtype)0., neuron_offset_diff);

		}

		// Compute derivative to bottom
		if (propagate_down[0]) {
			for (int s=0; s<sums_; ++s) {
				ComputeBottomDiffSum<Dtype><<<CAFFE_GET_BLOCKS(M_*K_), CAFFE_CUDA_NUM_THREADS>>>(M_*K_, s, bottom_diff, bottom_data, neuron_weight, maxs_data, top_diff, sums_, K_);
				CUDA_POST_KERNEL_CHECK;
			}
		}
    //static bool flag = false;
    // Regularise by variance of the weight across different APL units
    if(this->layer_param_.apl_param().layer_weight_decay() > 1e-9){
      string type = this->layer_param_.apl_param().slope_regular_type();
      if (type == string("Variance")){
        Variance_regularise_gpu(neuron_weight);
      } else if (type == string("L2")){
        L2_regularise_gpu(neuron_weight);
      } else{
        LOG(ERROR) << "Unsupported type: " << type 
            << ". Should be one of [\"Variance\", \"L2\"]";
      }
      caffe_gpu_add(temp1_.count(), neuron_weight_diff, temp1_.gpu_data(),
                neuron_weight_diff);
      
/*      if (!flag){
        LOG(INFO) << "input.txt";
        DumpMatrixToTxt("/home/xuzhenqi/input.txt", temp1_.count(), this->blobs_[0]->cpu_data(), temp1_.shape());
        DumpMatrixToTxt("/home/xuzhenqi/output.txt", temp1_.count(), temp1_.cpu_data(), temp1_.shape());
      }*/
      Variance_regularise_gpu(neuron_offset);
      caffe_gpu_add(temp1_.count(), neuron_offset_diff, temp1_.gpu_data(),
                neuron_offset_diff);
     /* if (!flag){
        DumpMatrixToTxt("/home/xuzhenqi/input1.txt", temp1_.count(), this->blobs_[1]->cpu_data(), temp1_.shape());
        DumpMatrixToTxt("/home/xuzhenqi/output1.txt", temp1_.count(), temp1_.cpu_data(), temp1_.shape());
      }
      flag = true;*/
    } 

	}

/**
 * Using temp1_ and temp2_ as buffer
 * The outcome will be stored in temp1_.gpu_data()
 */
template <class Dtype>
void APLLayer<Dtype>::Variance_regularise_gpu(const Dtype* weight){
      caffe_gpu_gemv<Dtype>(CblasNoTrans, sums_, K_, 1./K_, weight, 
        ones_.gpu_data(), 0., temp1_.mutable_gpu_data()); // Ex
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, sums_, K_, 1, 
          Dtype(1.), temp1_.gpu_data(), ones_.gpu_data(), Dtype(0),
          temp2_.mutable_gpu_data()); 
      caffe_gpu_sub(temp2_.count(), weight, temp2_.gpu_data(), 
                temp1_.mutable_gpu_data()); // X - Ex
      caffe_gpu_powx(temp1_.count(), weight, Dtype(2.), 
                 temp2_.mutable_gpu_data()); // x^2
      caffe_gpu_sub(temp1_.count(), ones_.gpu_data(), temp2_.gpu_data(), 
                temp2_.mutable_gpu_data()); // 1 - x^2
      caffe_gpu_mul(temp1_.count(), temp1_.gpu_data(), temp2_.gpu_data(), 
                temp2_.mutable_gpu_data()); // (1 - x^2)(X - Ex)
      caffe_gpu_powx(temp1_.count(), temp1_.gpu_data(), Dtype(2.),
                 temp1_.mutable_gpu_data()); // (x - Ex) ^ 2
      caffe_gpu_mul(temp1_.count(), temp1_.gpu_data(), weight,
                temp1_.mutable_gpu_data()); // x * (x - Ex)
      caffe_gpu_gemv<Dtype>(CblasNoTrans, sums_, K_, 1./K_, 
          temp2_.gpu_data(), ones_.gpu_data(), 0., 
          temp2_.mutable_gpu_diff()); // E((1-x^2)(x-Ex))
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, sums_, K_,
          1, Dtype(1.), temp2_.gpu_diff(), ones_.gpu_data(), Dtype(0),
          temp1_.mutable_gpu_diff());
      caffe_gpu_add(temp1_.count(), temp1_.gpu_diff(), temp1_.gpu_data(),
                temp1_.mutable_gpu_data());
      caffe_gpu_sub(temp1_.count(), temp1_.gpu_data(), temp2_.gpu_data(), 
                    temp1_.mutable_gpu_data());
      Dtype weight_decay = this->layer_param_.apl_param().layer_weight_decay();
      caffe_gpu_scal(temp1_.count(), weight_decay, temp1_.mutable_gpu_data());
}

/**
 * Using temp1_ as buffer
 * The outcome will be stored in temp1_.gpu_data()
 */
template <class Dtype>
void APLLayer<Dtype>::L2_regularise_gpu(const Dtype* weight){
  Dtype weight_decay = this->layer_param_.apl_param().layer_weight_decay();
  caffe_gpu_axpby(temp1_.count(), weight_decay, weight, Dtype(0.), temp1_.mutable_gpu_data());
}

INSTANTIATE_LAYER_GPU_FUNCS(APLLayer);

}  // namespace caffe
