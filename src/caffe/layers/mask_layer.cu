#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, K_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
  }
}

template <typename Dtype>
void MaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, K_, (Dtype)1.,
        top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
  }

  // Make the weights matrix diagonal
  int N = K_*K_;
  Blob<Dtype>* weights = this->blobs_[0].get();
  caffe_gpu_mul<Dtype>(N, weights->gpu_data(), identity->gpu_data(), weights->mutable_gpu_data());
}

INSTANTIATE_LAYER_GPU_FUNCS(MaskLayer);

}  // namespace caffe
