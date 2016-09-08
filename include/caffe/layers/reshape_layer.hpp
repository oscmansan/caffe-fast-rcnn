#ifndef CAFFE_XXX_LAYER_HPP_
#define CAFFE_XXX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief Reshapes the input Blob into an arbitrary-sized output Blob.
 *
 * Note: similarly to FlattenLayer, this layer does not change the input values
 * (see FlattenLayer, Blob::ShareData and Blob::ShareDiff).
 */
template <typename Dtype, typename Mtype>
class ReshapeLayer : public Layer<Dtype,Mtype> {
 public:
  explicit ReshapeLayer(const LayerParameter& param)
      : Layer<Dtype,Mtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top);

  virtual inline const char* type() const { return "Reshape"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top) {}
  virtual void Backward_cpu(const vector<Blob<Dtype,Mtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {}
  virtual void Forward_gpu(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top) {}
  virtual void Backward_gpu(const vector<Blob<Dtype,Mtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {}

  /// @brief vector of axes indices whose dimensions we'll copy from the bottom
  vector<int> copy_axes_;
  /// @brief the index of the axis whose dimension we infer, or -1 if none
  int inferred_axis_;
  /// @brief the product of the "constant" output dimensions
  int constant_count_;
};

}  // namespace caffe

#endif  // CAFFE_XXX_LAYER_HPP_
