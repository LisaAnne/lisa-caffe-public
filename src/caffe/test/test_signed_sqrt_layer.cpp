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
#include "caffe/layers/signed_sqrt_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class SignedSqrtLayerTest: public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

 protected:
    SignedSqrtLayerTest() :
            blob_bottom_0_(new Blob<Dtype>(2, 3, 6, 5)),
            blob_top_(new Blob<Dtype>()) {}
    virtual void SetUp() {
        // fill the values
        FillerParameter filler_param;
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_0_);
        // try to avoid testing in the unstable region
        caffe_add_scalar(blob_bottom_0_->count(), Dtype(3.0),
                         blob_bottom_0_->mutable_cpu_data());

        blob_bottom_vec_.push_back(blob_bottom_0_);
        blob_top_vec_.push_back(blob_top_);
    }

    virtual ~SignedSqrtLayerTest() {
        delete blob_bottom_0_;
        delete blob_top_;
    }

    Blob<Dtype>* const blob_bottom_0_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SignedSqrtLayerTest, TestDtypesAndDevices);

TYPED_TEST(SignedSqrtLayerTest, TestGradientOutplace) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    SignedSqrtLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-1);
    checker.CheckGradient(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
}

}  // namespace caffe
