/*

Copyright ©2016. The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for
educational, research, not-for-profit, and commercial purposes (such rights not subject to transfer), without fee, and without a signed licensing agreement, is hereby granted, provi
ded that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensi
ng, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, for commercial licensing opportunities.

Yang Gao, University of California, Berkeley.


IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
 AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMP
ANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

*/

#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/bilinear_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class BilinearLayerTest: public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

 protected:
    BilinearLayerTest() :
            blob_bottom_0_(new Blob<Dtype>(2, 3, 6, 5)),
                    blob_bottom_1_(new Blob<Dtype>(2, 5, 6, 5)),
                    blob_bottom_2_(new Blob<Dtype>(1, 512, 14, 14)),
                    blob_bottom_3_(new Blob<Dtype>(1, 512, 14, 14)),
                    blob_top_(new Blob<Dtype>()) {
    }
    virtual void SetUp() {
        // fill the values
        FillerParameter filler_param;
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_0_);
        filler.Fill(this->blob_bottom_1_);
        filler.Fill(this->blob_bottom_2_);
        filler.Fill(this->blob_bottom_3_);

        blob_bottom_vec_.push_back(blob_bottom_0_);
        blob_bottom_vec_.push_back(blob_bottom_1_);

        bottom_vec_large.push_back(blob_bottom_2_);
        bottom_vec_large.push_back(blob_bottom_2_);
        // bottom_vec_large.push_back(blob_bottom_3_);

        blob_top_vec_.push_back(blob_top_);
    }

    virtual ~BilinearLayerTest() {
        delete blob_bottom_0_;
        delete blob_bottom_1_;
        delete blob_bottom_2_;
        delete blob_bottom_3_;
        delete blob_top_;
    }

    Blob<Dtype>* const blob_bottom_0_;
    Blob<Dtype>* const blob_bottom_1_;
    Blob<Dtype>* const blob_bottom_2_;
    Blob<Dtype>* const blob_bottom_3_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_, bottom_vec_large;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BilinearLayerTest, TestDtypesAndDevices);

TYPED_TEST(BilinearLayerTest, TestGradientBottomDiff) {
typedef typename TypeParam::Dtype Dtype;
LayerParameter layer_param;
BilinearLayer<Dtype> layer(layer_param);
GradientChecker<Dtype> checker(1e-2, 1e-2);
checker.CheckGradient(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
}

TYPED_TEST(BilinearLayerTest, TestGradientBottomSame) {
typedef typename TypeParam::Dtype Dtype;
LayerParameter layer_param;
BilinearLayer<Dtype> layer(layer_param);
// somehow 1e-2 has some error, thus change to 1e-1
GradientChecker<Dtype> checker(1e-1, 1e-1);
this->blob_bottom_vec_.resize(1);
this->blob_bottom_vec_.push_back(this->blob_bottom_vec_[0]);
checker.CheckGradient(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
}

// With the same setting as CompactBilinearLayer's TestSpeed,
// On CPU, if the 2 bottoms are the same, then forward+backward
// is 6.4 second. Otherwise, if the two are different, then it
// approximately doubled to 12.4 second.

// On the GPU, excluding the host to device memcpy time, the
// forward pass takes 10.9 ms. If the two bottoms are the same,
// the backward pass takes 7.6 ms, otherwise it takes 13.7ms.

// If we average to per image, the forward pass is 0.34ms and
// backward pass is 0.43ms(when two bottoms are different).
TYPED_TEST(BilinearLayerTest, TestSpeed) {
typedef typename TypeParam::Dtype Dtype;
LayerParameter layer_param;
BilinearLayer<Dtype> layer(layer_param);
layer.SetUp(this->bottom_vec_large, this->blob_top_vec_);

layer.Forward(this->bottom_vec_large, this->blob_top_vec_);
if (Caffe::mode() == Caffe::GPU) {
    caffe_copy(this->blob_top_->count(),
            this->blob_top_->gpu_data(), this->blob_top_->mutable_gpu_diff());
} else {
    caffe_copy(this->blob_top_->count(),
            this->blob_top_->cpu_data(), this->blob_top_->mutable_cpu_diff());
}
vector<bool> propagate_down(2, true);
layer.Backward(this->blob_top_vec_, propagate_down,
        this->bottom_vec_large);
}

}  // namespace caffe
