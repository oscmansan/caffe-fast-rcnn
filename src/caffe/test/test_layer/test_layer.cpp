#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
using namespace std;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/get.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#endif
using namespace caffe;

class LayerTest {
public:
    LayerTest() {
        Caffe::set_mode(Caffe::GPU);
        init();
    }

    void ConvolutionLayerTest() {
        // Fill bottom blob
        vector<int> shape {1, 1, 7, 5};
        bottom_blob->Reshape(shape);
        init_rand(bottom_blob); print_blob(bottom_blob);

        // Set up layer parameters
        LayerParameter layer_param;
        ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
        conv_param->add_kernel_size(3);
        conv_param->add_stride(2);
        conv_param->set_num_output(4); // number of filters
        conv_param->mutable_weight_filler()->set_type("constant"); // type of filters
        conv_param->mutable_weight_filler()->set_value(1.0);

        // Create layer
        std::shared_ptr<Layer<float16,float16> > layer(new ConvolutionLayer<float16,float16>(layer_param));
        layer->SetUp(bottom,top);

        assert(top_blob->num() == 1);
        assert(top_blob->channels() == 4);
        assert(top_blob->height() == 3);
        assert(top_blob->width() == 2);

        Blob<float16,float16>* weights = layer->blobs()[0].get();
        print_blob(weights);

        // Run forward pass
        layer->Forward(bottom,top);
        print_blob(top_blob);
    }

private:
    Blob<float16,float16>* bottom_blob;
    Blob<float16,float16>* top_blob;
    vector<Blob<float16,float16>*> bottom;
    vector<Blob<float16,float16>*> top;

    void init() {
        // Create bottom blob
        bottom_blob = new Blob<float16,float16>();
        bottom.push_back(bottom_blob);
        
        // Create top blob
        top_blob = new Blob<float16,float16>();
        top.push_back(top_blob);
    }

    void print_shape(Blob<float16,float16>* blob) {
        vector<int> shape = blob->shape();
        cout << "(" << shape[0];
        for (int i = 1; i < shape.size(); ++i) {
            cout << "," << shape[i];
        }
        cout << ")" << endl;
    }

    void print_blob(Blob<float16,float16>* blob) {
        print_shape(blob);
        const float16* data = blob->cpu_data();
        vector<int> shape = blob->shape();
        for (int i = 0; i < shape[0]*shape[1]; ++i) {
            for (int j = 0; j < shape[2]; ++j) {
                for (int k = 0; k < shape[3]; ++k) {
                    cout << data[i*shape[2]*shape[3]+j*shape[3]+k] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }

    void init_rand(Blob<float16,float16>* blob) {
        float16* data = blob->mutable_cpu_data();
        for (int i = 0; i < blob->count(); ++i) {
            data[i] = Get<float16>(float(rand())/float(RAND_MAX));
        }
    }

    void init_ones(Blob<float16,float16>* blob) {
        float16* data = blob->mutable_cpu_data();
        for (int i = 0; i < blob->count(); ++i) {
            data[i] = Get<float16>(1.);
        }
    }
};


int main() {
    LayerTest test;
    test.ConvolutionLayerTest();
}
