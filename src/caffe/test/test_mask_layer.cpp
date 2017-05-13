#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class MaskLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MaskLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_nobatch_(new Blob<Dtype>(1, 2, 3, 4)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MaskLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_nobatch_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_nobatch_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MaskLayerTest, TestDtypesAndDevices);

TYPED_TEST(MaskLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  shared_ptr<MaskLayer<Dtype> > layer(
      new MaskLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 3*4*5);
}

TYPED_TEST(MaskLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  FillerParameter* filler_param = 
      inner_product_param->mutable_weight_filler();
  filler_param->set_type("uniform");
  filler_param->set_value(1.);
  vector<bool> bottom_need_backward;
  bottom_need_backward.resize(this->blob_bottom_vec_.size());
  for (int i = 0; i < bottom_need_backward.size(); ++i) {
    bottom_need_backward[i] = true;
  }

  shared_ptr<MaskLayer<Dtype> > layer(
        new MaskLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Backward(this->blob_bottom_vec_, bottom_need_backward, this->blob_top_vec_);

  const Dtype* data = layer->blobs()[0]->cpu_data();
  const int count = layer->blobs()[0]->count();
  for (int i = 0; i < count; ++i) {
      std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

}  // namespace caffe
