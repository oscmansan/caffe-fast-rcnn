#include <cstring>
#include <vector>
#include <google/protobuf/text_format.h>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

typedef ::testing::Types<GPUDevice<float>, GPUDevice<double> > TestDtypesAndDevicesGPUOnly;

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

TYPED_TEST_CASE(MaskLayerTest, TestDtypesAndDevicesGPUOnly);

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

TYPED_TEST(MaskLayerTest, TestForwardBackward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  FillerParameter* filler_param = 
      inner_product_param->mutable_weight_filler();
  //filler_param->set_type("constant");
  //filler_param->set_value(1.);
  filler_param->set_type("uniform");
  vector<bool> bottom_need_backward;
  bottom_need_backward.resize(this->blob_bottom_vec_.size());
  for (int i = 0; i < bottom_need_backward.size(); ++i) {
    bottom_need_backward[i] = true;
  }

  shared_ptr<MaskLayer<Dtype> > layer(
        new MaskLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Backward(this->blob_bottom_vec_, bottom_need_backward, this->blob_top_vec_);

  Blob<Dtype>* weights = layer->blobs()[0].get();
  const Dtype* data = weights->cpu_data();
  vector<int> shape = weights->shape();
  EXPECT_EQ(shape[0], shape[1]);
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      Dtype x = data[i*shape[0]+j];
      if (i == j) {
        EXPECT_GE(x, 0);
      }
      else {
        EXPECT_EQ(x, 0);
      }
    }
  }
}

TYPED_TEST(MaskLayerTest, TestReadLayerParamsFromFile) { 
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);

  string layer_string = 
    "name: \"fc6_mask\" \
     type: \"Mask\" \
     bottom: \"fc6\" \
     top: \"fc6\" \
     param { \
       lr_mult: 1 \
     } \
     inner_product_param { \
       weight_filler { \
         type: \"constant\" \
         value: 1 \
       } \
     }";  
  LayerParameter layer_param;
  google::protobuf::TextFormat::ParseFromString(layer_string, &layer_param);

  shared_ptr<MaskLayer<Dtype> > layer(new MaskLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Blob<Dtype>* weights = layer->blobs()[0].get();
  const Dtype* data = weights->cpu_data();
  vector<int> shape = weights->shape();
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      Dtype x = data[i*shape[0]+j];
      if (i == j) {
        EXPECT_EQ(x, 1);
      }
      else {
        EXPECT_EQ(x, 0);
      }
    }
  }
}

}  // namespace caffe
