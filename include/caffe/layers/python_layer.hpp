#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;

namespace caffe {

template <typename Dtype, typename Mtype>
class PythonLayer : public Layer<Dtype,Mtype> {
 public:
  PythonLayer(PyObject* self, const LayerParameter& param)
      : Layer<Dtype,Mtype>(param), self_(bp::handle<>(bp::borrowed(self))) { }

  virtual void LayerSetUp(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top) {
    // Disallow PythonLayer in MultiGPU training stage, due to GIL issues
    // Details: https://github.com/BVLC/caffe/issues/2936
    if (this->phase_ == TRAIN && Caffe::solver_count() > 1
        && !ShareInParallel()) {
      LOG(FATAL) << "PythonLayer is not implemented in Multi-GPU training";
    }
    self_.attr("param_str_") = bp::str(
        this->layer_param_.python_param().param_str());
    self_.attr("setup")(bottom, top);
  }
  virtual void Reshape(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top) {
    self_.attr("reshape")(bottom, top);
  }

  virtual inline bool ShareInParallel() const {
    return this->layer_param_.python_param().share_in_parallel();
  }

  virtual inline const char* type() const { return "Python"; }
  
  void SetUp(const vector<Blob<float16,float16>*>& bottom,
      const vector<Blob<float16,float16>*>& top) {

      vector<Blob<Dtype,Mtype>*> bottom2;
      vector<Blob<Dtype,Mtype>*> top2;

      // float16 -> Dtype conversion
      for (int i = 0; i < bottom.size(); ++i) {
          bottom2.push_back(new Blob<Dtype,Mtype>());
      }
      for (int i = 0; i < top.size(); ++i) {
          top2.push_back(new Blob<Dtype,Mtype>());
      }

      this->LayerSetUp(bottom2,top2);

      // Dtype -> float16 conversion
      for (int i = 0; i < top2.size(); ++i) {
          top[i]->Reshape(top2[i]->shape());
      }
  }
  
  void Forward(const vector<Blob<float16,float16>*>& bottom,
          const vector<Blob<float16,float16>*>& top) {

      vector<Blob<Dtype,Mtype>*> bottom2;
      vector<Blob<Dtype,Mtype>*> top2;

      // float16 -> Dtype conversion
      for (int i = 0; i < bottom.size(); ++i) {
          Blob<float16,float16>* blob = bottom[i];
          Blob<Dtype,Mtype>* blob2 = new Blob<Dtype,Mtype>(bottom[i]->shape());
          float16* data = blob->mutable_cpu_data();
          Dtype* data2 = blob2->mutable_cpu_data();
          for (int j = 0; j < blob->count(); ++j) {
              data2[j] = Get<Dtype>(data[j]);
          }
          bottom2.push_back(blob2);
      }
      for (int i = 0; i < top.size(); ++i) {
          top2.push_back(new Blob<Dtype,Mtype>(top[i]->shape()));
      }

      this->Forward_cpu(bottom2,top2);

      // Dtype -> float16 conversion
      for (int i = 0; i < top2.size(); ++i) {
          Blob<float16,float16>* blob = top[i];
          Blob<Dtype,Mtype>* blob2 = top2[i];
          blob->Reshape(blob2->shape());
          float16* data = blob->mutable_cpu_data();
          Dtype* data2 = blob2->mutable_cpu_data();
          for (int j = 0; j < blob->count(); ++j) {
              data[j] = Get<float16>(data2[j]);
          }
      }
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top) {
    self_.attr("forward")(bottom, top);
  }

  virtual void Forward_gpu(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top) {
      Forward_cpu(bottom,top);
  }

  virtual void Backward_cpu(const vector<Blob<Dtype,Mtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {
    self_.attr("backward")(top, propagate_down, bottom);
  }

 private:
  bp::object self_;
};

}  // namespace caffe

#endif
