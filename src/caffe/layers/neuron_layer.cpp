#include <vector>

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void NeuronLayer<Dtype,Mtype>::Reshape(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
