#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(model, "",
              "The model definition protocol buffer text file..");
DEFINE_int32(iterations, 100,
             "The number of iterations to run.");

void time() {
  Net<float> net(FLAGS_model, caffe::TEST);
  net.ForwardFrom(0);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";
  const vector<shared_ptr<Layer<float> > >& layers = net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      net.bottom_need_backward();
  std::vector<double> time_per_layer(layers.size(), 0.0);
  Timer total_timer, timer;
  total_timer.Start();
  for (int j = 0; j < FLAGS_iterations; ++j) {
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      time_per_layer[i] += timer.MicroSeconds();
    }
  }
  total_timer.Stop();
  LOG(INFO) << "Average time: " << total_timer.MilliSeconds() /
      FLAGS_iterations << " ms.";
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(20) << layername <<
        "\t" << time_per_layer[i] / 1000 / FLAGS_iterations << " ms.";
  }
}


int main(int argc, char**argv) {
  FLAGS_alsologtostderr = 1;
  caffe::GlobalInit(&argc, &argv);

  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score";
  Caffe::set_mode(Caffe::CPU);
  time();
  return 0;
}
