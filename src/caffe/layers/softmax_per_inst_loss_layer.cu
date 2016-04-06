#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxPerInstLossForward(const int nthreads,
          const Dtype* prob_data, const Dtype* label,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_index = n * spatial_dim + s;
    const int label_value = static_cast<int>(label[label_index]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[label_index] = 0;
    } else {
      const int prob_index = n * dim + label_value * spatial_dim + s;
      loss[label_index] = -log(max(prob_data[prob_index], Dtype(FLT_MIN)));
    }
  }
}

template <typename Dtype>
void SoftmaxPerInstLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  Dtype* loss = top[0]->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxPerInstLossForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, loss);
}

template <typename Dtype>
__global__ void SoftmaxPerInstLossBackward(const int nthreads,
          const Dtype* top, const Dtype* label, const Dtype* top_diff,
          const Dtype* prob_data, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* bottom_diff) {
  const int channels = dim / spatial_dim;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_index = n * spatial_dim + s;
    const int label_value = static_cast<int>(label[label_index]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
    } else {
      for (int c = 0; c < channels; ++c) {
        const int index = n * dim + c * spatial_dim + s;
        bottom_diff[index] =
            top_diff[label_index] * (prob_data[index] - (c == label_value));
      }
    }
  }
}

template <typename Dtype>
void SoftmaxPerInstLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
//   if (propagate_down[1]) {
//     LOG(FATAL) << this->type()
//                << " Layer cannot backpropagate to label inputs.";
//   }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    const Dtype* top_diff = top[0]->gpu_diff();
    SoftmaxPerInstLossBackward<Dtype>
        // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, top_data, label, top_diff, prob_data,
        outer_num_, dim, inner_num_,
        has_ignore_label_, ignore_label_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxPerInstLossLayer);

}  // namespace caffe
