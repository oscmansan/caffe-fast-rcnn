#include <iostream>
#include <vector>
#include <string>
#include <time.h>
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

#define Dtype float16
#define Mtype float16

class LayerTest {
public:
    LayerTest() {
        Caffe::set_mode(Caffe::GPU);
        init();
    }

    void ConvolutionLayerTest() {
        // Fill bottom blob
        vector<int> shape {1000, 3, 256, 256};
        bottom_blob->Reshape(shape);
        init_rand(bottom_blob); 
        print_shape(bottom_blob);

        // Set up layer parameters
        LayerParameter layer_param;
        ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
        conv_param->add_kernel_size(3);
        conv_param->add_stride(2);
        conv_param->set_num_output(4); // number of filters
        conv_param->mutable_weight_filler()->set_type("constant"); // type of filters
        conv_param->mutable_weight_filler()->set_value(1.0);

        // Create layer
        std::shared_ptr<Layer<Dtype,Mtype> > layer(new ConvolutionLayer<Dtype,Mtype>(layer_param));
        layer->SetUp(bottom,top);

        assert(top_blob->num() == 1000);
        assert(top_blob->channels() == 4);
        assert(top_blob->height() == 127);
        assert(top_blob->width() == 127);

        Blob<Dtype,Mtype>* weights = layer->blobs()[0].get();
        print_shape(weights);

        // Run forward pass
        timespec start,end,elapsed;
        clock_gettime(CLOCK_REALTIME,&start);
        layer->Forward(bottom,top);
        clock_gettime(CLOCK_REALTIME,&end);
        print_shape(top_blob);
        elapsed = diff(start,end);
        cout<<elapsed.tv_sec*1000000000+elapsed.tv_nsec<<endl;
    }

private:
    Blob<Dtype,Mtype>* bottom_blob;
    Blob<Dtype,Mtype>* top_blob;
    vector<Blob<Dtype,Mtype>*> bottom;
    vector<Blob<Dtype,Mtype>*> top;

    void init() {
        // Create bottom blob
        bottom_blob = new Blob<Dtype,Mtype>();
        bottom.push_back(bottom_blob);
        
        // Create top blob
        top_blob = new Blob<Dtype,Mtype>();
        top.push_back(top_blob);
    }

    void print_shape(Blob<Dtype,Mtype>* blob) {
        vector<int> shape = blob->shape();
        cout << "(" << shape[0];
        for (int i = 1; i < shape.size(); ++i) {
            cout << "," << shape[i];
        }
        cout << ")" << endl << flush;
    }

    void print_blob(Blob<Dtype,Mtype>* blob) {
        print_shape(blob);
        const Dtype* data = blob->cpu_data();
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
        cout << endl << flush;
    }

    void init_rand(Blob<Dtype,Mtype>* blob) {
        Dtype* data = blob->mutable_cpu_data();
        for (int i = 0; i < blob->count(); ++i) {
            data[i] = Get<float16>(float(rand())/float(RAND_MAX));
        }
    }

    void init_ones(Blob<Dtype,Mtype>* blob) {
        Dtype* data = blob->mutable_cpu_data();
        for (int i = 0; i < blob->count(); ++i) {
            data[i] = Get<float16>(1.);
        }
    }

    timespec diff(timespec start, timespec end) {
        timespec temp;
        if ((end.tv_nsec-start.tv_nsec)<0) {
            temp.tv_sec = end.tv_sec-start.tv_sec-1;
            temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
        } 
        else {
            temp.tv_sec = end.tv_sec-start.tv_sec;
            temp.tv_nsec = end.tv_nsec-start.tv_nsec;
       }
       return temp;
    }
};


int main() {
    LayerTest test;
    test.ConvolutionLayerTest();
}
