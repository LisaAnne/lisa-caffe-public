#include <cfloat>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void SampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->shape(0);
  vector<int> top_shape(1, num);
  rand_.Reshape(top_shape);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void SampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->shape(0);
  const int dim = bottom[0]->count() / top[0]->count();
  if (this->phase_ == TRAIN) {
    Dtype* rand_data = rand_.mutable_cpu_data();
    caffe_set(num, Dtype(-1), top_data);
    caffe_rng_uniform<Dtype>(rand_.count(), Dtype(0), Dtype(1), rand_data);
    for (int i = 0; i < num; ++i) {
      const Dtype r = rand_data[i];
      Dtype cum_sum = 0;
      for (int j = 0; j < dim; ++j) {
        cum_sum += bottom_data[j];
        if (cum_sum >= r) {
          top_data[i] = static_cast<Dtype>(j);
          break;
        }
      }
      bottom_data += dim;
    }
  } else {
    for (int i = 0; i < num; ++i) {
      int argmax = -1;
      Dtype max = -FLT_MAX;
      for (int j = 0; j < dim; ++j) {
        const Dtype datum = bottom_data[j];
        if (datum > max) {
          argmax = j;
          max = datum;
        }
      }
      top_data[i] = argmax;
      bottom_data += dim;
    }
  }
}

INSTANTIATE_CLASS(SampleLayer);
REGISTER_LAYER_CLASS(Sample);

}  // namespace caffe
