#ifndef CAFFE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype, typename Mtype>
class InnerProductLayer : public Layer<Dtype,Mtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype,Mtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top);

  virtual inline const char* type() const { return "InnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype,Mtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype,Mtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype,Mtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
