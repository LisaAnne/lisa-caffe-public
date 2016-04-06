#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxPerInstLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
}

template <typename Dtype>
void SoftmaxPerInstLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "The data and label should have the same number.";
  top[0]->ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void SoftmaxPerInstLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  Dtype* loss = top[0]->mutable_cpu_data();
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j, ++label, ++loss) {
      const int label_value = static_cast<int>(*label);
      if (has_ignore_label_ && label_value == ignore_label_) {
        *loss = 0;
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      const int prob_index = i * dim + label_value * inner_num_ + j;
      *loss = -log(std::max(prob_data[prob_index], Dtype(FLT_MIN)));
    }
  }
}

template <typename Dtype>
void SoftmaxPerInstLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//   if (propagate_down[1]) {
//     LOG(FATAL) << this->type()
//                << " Layer cannot backpropagate to label inputs.";
//   }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    const Dtype* loss_weight = top[0]->cpu_diff();
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j, ++label, ++loss_weight) {
        const int label_value = static_cast<int>(*label);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            const int index = i * dim + c * inner_num_ + j;
            bottom_diff[index] =
                (*loss_weight) * (prob_data[index] - (c == label_value));
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxPerInstLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxPerInstLossLayer);
REGISTER_LAYER_CLASS(SoftmaxPerInstLoss);

}  // namespace caffe
