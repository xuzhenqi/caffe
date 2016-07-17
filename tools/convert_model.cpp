#include <algorithm>
#include <vector>
#include <string>

#include <tr1/unordered_map>
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"

using namespace caffe;
typedef std::tr1::unordered_map<std::string, std::string> umap;

#define var_eps_ float(1e-10)

typedef void (*integ_inner)(const float *scale_data, const float *shift_data,
                            float *w, float *b, int num_input, int num_output);
void process_bn(float *scale_data, float *shift_data,
                const float *his_mean_data, const float *his_var_data, int _channels);
bool integrate(std::tr1::unordered_map<string, int> &bn_bottom2index, int blob_id,
               const string &blob_name, NetParameter &src_net_param,
               LayerParameter *dst_param, integ_inner fun);

void integrate_ip(const float *scale_data, const float *shift_data,
                  float *w, float *b, int num_input, int num_output) {
	caffe_cpu_gemm(CblasNoTrans, CblasNoTrans,
	               num_input, 1, num_output, float(1),
	               w, shift_data, float(1), b);
	for (int c = 0; c < num_input; ++c)
		caffe_mul(num_output, w + c * num_output, scale_data, w + c * num_output);
}

void integrate_conv_before(const LayerParameter& bn_param, LayerParameter& conv_param) {
	float* conv_weight=conv_param.mutable_blobs(0)->mutable_data()->mutable_data();
	float* conv_bias=conv_param.mutable_blobs(1)->mutable_data()->mutable_data();
	const float* bn_scale=bn_param.blobs(0).data().data();
	const float* bn_shift=bn_param.blobs(1).data().data();
	int in_channels=conv_param.blobs(0).shape().dim(1);
	int out_channels=conv_param.blobs(0).shape().dim(0);
	int spatial_stride=conv_param.blobs(0).shape().dim(2)*conv_param.blobs(0).shape().dim(3);
	int kernel_stride=spatial_stride*in_channels;
	vector<float> bn_shift_multi(kernel_stride);
	for(size_t i=0;i<bn_shift_multi.size();i++) {
		bn_shift_multi[i]=bn_shift[i/spatial_stride];
	}
	for(int i=0;i<out_channels;i++) {
		conv_bias[i]+=caffe_cpu_dot(kernel_stride, bn_shift_multi.data(), conv_weight+kernel_stride*i);
	}
	for(int i=0;i<out_channels;i++) {
		for(int j=0;j<in_channels;j++) {
			caffe_scal(spatial_stride, bn_scale[j], conv_weight+spatial_stride*(i*in_channels+j));
		}
	}
}

void integrate_conv(const float *scale_data, const float *shift_data,
                    float *w, float *b, int num_input, int num_output) {
	for (int c = 0; c < num_output; ++c)
		caffe_scal(num_input, scale_data[c], w + c * num_input);
	caffe_mul(num_output, b, scale_data, b);
	caffe_add(num_output, b, shift_data, b);
}

int main(int argc, char** argv) {
	if (argc != 3 && argc != 4) {
		LOG(ERROR) << "net_proto_file_in net_proto_file_out [--remove_param]";
		return 1;
	}
	bool rm_param = argc == 4;
    
    Caffe::set_mode(Caffe::CPU);

	NetParameter src_net_param;
	ReadNetParamsFromBinaryFileOrDie(argv[1], &src_net_param);
	int _layers_size = src_net_param.layer_size();
	LOG(INFO) << "Total layers: " << _layers_size;
	LOG(INFO) << "Recompute BN parameter...";
	std::tr1::unordered_map<string, int> bn_bottom2index;
	umap bn_top2bottom;
	for (int i = 0; i < _layers_size; ++i) {
		LayerParameter * current_param = src_net_param.mutable_layer(i);
		string layer_type = current_param->type();
		std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::tolower);
		if (layer_type.find("bn") != string::npos) {
			CHECK_EQ(current_param->bottom_size(), 1) << "BN layer is only allowed to have one bottom!";
			LOG(INFO) << "BN input blob: " << current_param->bottom(0);
			bn_bottom2index[current_param->bottom(0)] = i;
			bn_top2bottom[current_param->top(0)] = current_param->bottom(0);

			float* scale_data = current_param->mutable_blobs(0)->mutable_data()->mutable_data();
			float* shift_data = current_param->mutable_blobs(1)->mutable_data()->mutable_data();
			const float* his_mean_data = current_param->blobs(2).data().data();
			const float* his_var_data = current_param->blobs(3).data().data();
			process_bn(scale_data, shift_data, his_mean_data,
			           his_var_data, current_param->blobs(0).data_size() );
		}
	}

	// int conv1_1_index;
	// for(size_t i=0;i<_layers_size;i++) {
	// 	if(src_net_param.layer(i).name()=="conv1_1") {
	// 		conv1_1_index=i;
	// 		break;
	// 	}
	// }
	// int data_bn_index;
	// for(size_t i=0;i<_layers_size;i++) {
	// 	if(src_net_param.layer(i).name()=="data_bn") {
	// 		data_bn_index=i;
	// 		break;
	// 	}
	// }
	// integrate_conv_before(src_net_param.layer(data_bn_index), *(src_net_param.mutable_layer(conv1_1_index)));

	LOG(INFO) << "Integrate BN start...";
	NetParameter dst_net_param;
	umap c_map;
	for (int i = 0; i < _layers_size; ++i) {
		LayerParameter * dst_param = src_net_param.mutable_layer(i);
		if ( dst_param->bottom_size() == 0 ) continue;
		string layer_type = dst_param->type();
		std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::tolower);
		if ( layer_type.find("bn") != string::npos || layer_type == "dropout" ) continue;
		if ( layer_type == "split") {
			for (int j = 0; j < dst_param->top_size(); ++j)
        		c_map[dst_param->top(j)] = dst_param->bottom(0);
		} else if ( layer_type.find("convolution") != string::npos ) {
			if (integrate(bn_bottom2index, 0, dst_param->top(0), src_net_param, dst_param, integrate_conv))
				bn_bottom2index.erase(dst_param->top(0));
		} else if ( layer_type == "slgrnn" || layer_type == "sllstm" ) {
			string blob_name;
			if (src_net_param.layer(i - 1).type() == "Reverse")
				blob_name = src_net_param.layer(i - 1).bottom(1);
			else blob_name = dst_param->bottom(0);
			if (c_map.find(blob_name) != c_map.end())
				blob_name = c_map[blob_name];
			integrate(bn_bottom2index, 2, blob_name, src_net_param, dst_param, integrate_ip);
		} else if ( layer_type.find("innerproduct") != string::npos ) {
			// interage bn before
			integrate(bn_bottom2index, 0, dst_param->bottom(0), src_net_param, dst_param, integrate_ip);
			// interage bn after
			integrate(bn_bottom2index, 0, dst_param->top(0), src_net_param, dst_param, integrate_conv);
		}
		if (rm_param)
			dst_param->clear_param();
		for (int j = 0; j < dst_param->bottom_size(); ++j)
			if ( bn_top2bottom.find(dst_param->bottom(j)) != bn_top2bottom.end() )
				*dst_param->mutable_bottom(j) = bn_top2bottom[dst_param->bottom(j)];
		for (int j = 0; j < dst_param->top_size(); ++j)
			if ( bn_top2bottom.find(dst_param->top(j)) != bn_top2bottom.end() )
				*dst_param->mutable_top(j) = bn_top2bottom[dst_param->top(j)];
		dst_net_param.add_layer()->CopyFrom(*dst_param);
	}

	LOG(INFO) << "Snapshotting to " << argv[2];
	WriteProtoToBinaryFile(dst_net_param, argv[2]);
	return 0;
}

void process_bn(float *scale_data, float *shift_data,
                const float *his_mean_data, const float *his_var_data, int _channels) {
	float * batch_statistic_ptr = new float[_channels];
	/** compute statistic value: scale' = \gamma / \sqrt(Var(x) + \epsilon) **/
	// sqrt(var(x) + \epsilon)
	caffe_copy(_channels, his_var_data, batch_statistic_ptr);
	caffe_add_scalar(_channels, var_eps_, batch_statistic_ptr);
	caffe_powx(_channels, batch_statistic_ptr, float(0.5), batch_statistic_ptr);
	// \gamma / \sqrt(Var(x) + \epsilon)
	caffe_div(_channels, scale_data, batch_statistic_ptr, scale_data);

	/** compute statistic value: shift' = \beta - \mu * scale' **/
	caffe_mul(_channels, scale_data, his_mean_data, batch_statistic_ptr);

	// \beta - \mu * scale'
	caffe_sub(_channels, shift_data, batch_statistic_ptr, shift_data);
	delete batch_statistic_ptr;
}

bool integrate(std::tr1::unordered_map<string, int> &bn_bottom2index, int blob_id,
               const string &blob_name, NetParameter &src_net_param,
               LayerParameter *dst_param, integ_inner fun) {
	if ( bn_bottom2index.find(blob_name) == bn_bottom2index.end() ) {
		LOG(ERROR) << blob_name << " not found";
		return false;
	}
	const LayerParameter &src_param = src_net_param.layer(bn_bottom2index[blob_name]);
	LOG(INFO) << "Integrate " << src_param.name() << " into " << dst_param->name();
	int num_output = src_param.blobs(0).data_size();
	int num_input = dst_param->blobs(blob_id).data_size() / num_output;
	float * wx = dst_param->mutable_blobs(blob_id)->mutable_data()->mutable_data();
	// if bias_term == false
	if (blob_id == 0 &&
		( (dst_param->has_convolution_param() && !dst_param->convolution_param().bias_term()) ||
		  (dst_param->has_inner_product_param() && !dst_param->inner_product_param().bias_term())
		)) {
		BlobProto * bias = dst_param->add_blobs();
		bias->mutable_shape()->add_dim(num_output);
		for (int j = 0; j < num_output; ++j)
			bias->add_data(float(0));
		if (dst_param->has_convolution_param())
			dst_param->mutable_convolution_param()->clear_bias_term();
		else dst_param->mutable_inner_product_param()->clear_bias_term();
		ParamSpec * param = dst_param->add_param();
		param->set_lr_mult(float(2));
		param->set_decay_mult(float(0));
	}
	float * b = dst_param->mutable_blobs(1)->mutable_data()->mutable_data();
	const float* scale_data = src_param.blobs(0).data().data();
	const float* shift_data = src_param.blobs(1).data().data();
	fun(scale_data, shift_data, wx, b, num_input, num_output);
	return true;
}