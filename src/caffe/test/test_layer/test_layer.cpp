#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>
#include <assert.h>
using namespace std;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/cudnn_softmax_layer.hpp"
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
        init_rand(bottom_blob); 
        cout << "I: " << to_string(bottom_blob->shape()) << endl;
        clog << to_string(bottom_blob) << endl;

        // Set up layer parameters
        LayerParameter layer_param;
        ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
        conv_param->add_kernel_size(3);
        conv_param->add_stride(2);
        conv_param->set_num_output(4); // number of filters
        //conv_param->mutable_weight_filler()->set_type("constant"); // type of filters
        //conv_param->mutable_weight_filler()->set_value(1.0);
        //conv_param->mutable_weight_filler()->set_type("gaussian");

        // Create layer
        std::shared_ptr<Layer<Dtype,Mtype> > layer(new ConvolutionLayer<Dtype,Mtype>(layer_param));
        layer->SetUp(bottom,top);

        assert(top_blob->num() == num);
        assert(top_blob->channels() == 4);
        assert(top_blob->height() == 1);
        assert(top_blob->width() == 2);

        Blob<Dtype,Mtype>* weights = layer->blobs()[0].get();
        init_rand(weights);
        cout << "W: " << to_string(weights->shape()) << endl;
        clog << to_string(weights) << endl;

        // Run forward pass
        timespec start,end,elapsed;
        clock_gettime(CLOCK_REALTIME,&start);
        layer->Forward(bottom,top);
        clock_gettime(CLOCK_REALTIME,&end);
        cout << "O: " << to_string(top_blob->shape()) << endl;
        clog << to_string(top_blob) << endl;
        elapsed = diff(start,end);
        cout<<"time: "<<elapsed.tv_sec*1000000000+elapsed.tv_nsec<<endl;
    }

    void InnerProductLayerTest() {
        // Fill bottom blob
        init_rand(bottom_blob);
        cout << "I: " << to_string(bottom_blob->shape()) << endl; 
        clog << to_string(bottom_blob) << endl;

        // Set up layer parameters
        LayerParameter layer_param;
        InnerProductParameter* inner_product_param = layer_param.mutable_inner_product_param();
        inner_product_param->set_num_output(10);
        //inner_product_param->mutable_weight_filler()->set_type("constant");
        //inner_product_param->mutable_weight_filler()->set_value(1.0);
        //inner_product_param->mutable_weight_filler()->set_type("uniform");

        // Create layer
        std::shared_ptr<Layer<Dtype,Mtype> > layer(new InnerProductLayer<Dtype,Mtype>(layer_param));
        layer->SetUp(bottom,top);

        assert(top_blob->num() == num);
        assert(top_blob->channels() == 10);
        assert(top_blob->height() == 1);
        assert(top_blob->width() == 1);

        Blob<Dtype,Mtype>* weights = layer->blobs()[0].get();
        init_rand(weights);
        cout << "W: " << to_string(weights->shape()) << endl;
        clog << to_string(weights) << endl;

        // Run forward pass
        timespec start,end,elapsed;
        clock_gettime(CLOCK_REALTIME,&start);
        layer->Forward(bottom,top);
        clock_gettime(CLOCK_REALTIME,&end);
        cout << "O: " << to_string(top_blob->shape()) << endl;
        clog << to_string(top_blob) << endl;
        elapsed = diff(start,end);
        cout<<"time: "<<elapsed.tv_sec*1000000000+elapsed.tv_nsec<<endl;
    }

    void PoolingLayerTest() {
        // Fill bottom blob
        init_rand(bottom_blob);
        cout << "I: " << to_string(bottom_blob->shape()) << endl; 
        clog << to_string(bottom_blob) << endl;

        // Set up layer parameters
        LayerParameter layer_param;
        PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
        pooling_param->set_kernel_size(2);
        pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);

        // Create layer
        std::shared_ptr<Layer<Dtype,Mtype> > layer(new PoolingLayer<Dtype,Mtype>(layer_param));
        layer->SetUp(bottom,top);

        assert(top_blob->num() == num);
        assert(top_blob->channels() == channels);
        assert(top_blob->height() == 3);
        assert(top_blob->width() == 4);

        // Run forward pass
        timespec start,end,elapsed;
        clock_gettime(CLOCK_REALTIME,&start);
        layer->Forward(bottom,top);
        clock_gettime(CLOCK_REALTIME,&end);
        cout << "O: " << to_string(top_blob->shape()) << endl;
        clog << to_string(top_blob) << endl;
        elapsed = diff(start,end);
        cout<<"time: "<<elapsed.tv_sec*1000000000+elapsed.tv_nsec<<endl;
    }

    void SplitLayerTest() {
        // Create second top blob
        Blob<Dtype,Mtype>* top_blob_2 = new Blob<Dtype,Mtype>();
        top.push_back(top_blob_2);

        // Fill bottom blob
        init_rand(bottom_blob);
        cout << "I: " << to_string(bottom_blob->shape()) << endl; 
        clog << to_string(bottom_blob) << endl;

        // Set up layer parameters
        LayerParameter layer_param;
        
        // Create layer
        SplitLayer<Dtype,Mtype> layer(layer_param);
        layer.SetUp(bottom,top);

        assert(top_blob->num() == num);
        assert(top_blob->channels() == channels);
        assert(top_blob->height() == height);
        assert(top_blob->width() == width);
        assert(top_blob_2->num() == num);
        assert(top_blob_2->channels() == channels);
        assert(top_blob_2->height() == height);
        assert(top_blob_2->width() == width);

        // Run forward pass
        timespec start,end,elapsed;
        clock_gettime(CLOCK_REALTIME,&start);
        layer.Forward(bottom,top);
        clock_gettime(CLOCK_REALTIME,&end);
        cout << "O: " << to_string(top_blob->shape()) << endl;
        cout << "O: " << to_string(top_blob_2->shape()) << endl;
        clog << to_string(top_blob) << endl;
        clog << to_string(top_blob_2) << endl;
        elapsed = diff(start,end);
        cout<<"time: "<<elapsed.tv_sec*1000000000+elapsed.tv_nsec<<endl;

        // Assert correct split
        for (int i = 0; i < bottom_blob->count(); ++i) {
            Dtype bottom_value = bottom_blob->cpu_data()[i];
            assert(top_blob->cpu_data()[i] == bottom_value);
            assert(top_blob_2->cpu_data()[i] == bottom_value);
        }    
    }

    void ReLULayerTest() {
        // Fill bottom blob
        FillerParameter filler_param;
        GaussianFiller<Dtype,Mtype> filler(filler_param);
        filler.Fill(bottom_blob);
        cout << "I: " << to_string(bottom_blob->shape()) << endl; 
        clog << to_string(bottom_blob) << endl;
           
        // Set up layer parameters
        LayerParameter layer_param;

        // Create layer
        ReLULayer<Dtype,Mtype> layer(layer_param);
        layer.SetUp(bottom,top);

        // Run forward pass
        timespec start,end,elapsed;
        clock_gettime(CLOCK_REALTIME,&start);
        layer.Forward(bottom,top);
        clock_gettime(CLOCK_REALTIME,&end);
        cout << "O: " << to_string(top_blob->shape()) << endl;
        clog << to_string(top_blob) << endl;
        elapsed = diff(start,end);
        cout<<"time: "<<elapsed.tv_sec*1000000000+elapsed.tv_nsec<<endl;
        
        // Check values
        const Dtype* bottom_data = bottom_blob->cpu_data();
        const Dtype* top_data = top_blob->cpu_data();
        for (int i = 0; i < bottom_blob->count(); ++i) {
            assert(top_data[i] >= 0.);
            assert(top_data[i] == 0 || top_data[i] == bottom_data[i]);
        }
    }

    void SoftmaxLayerTest() {
        // Fill bottom blob
        init_rand(bottom_blob);
        cout << "I: " << to_string(bottom_blob->shape()) << endl; 
        clog << to_string(bottom_blob) << endl;

        // Set up layer parameters
        LayerParameter layer_param;

        // Create layer
        CuDNNSoftmaxLayer<Dtype,Mtype> layer(layer_param);
        layer.SetUp(bottom,top);

        // Run forward pass
        timespec start,end,elapsed;
        clock_gettime(CLOCK_REALTIME,&start);
        layer.Forward(bottom,top);
        clock_gettime(CLOCK_REALTIME,&end);
        cout << "O: " << to_string(top_blob->shape()) << endl;
        clog << to_string(top_blob) << endl;
        elapsed = diff(start,end);
        cout<<"time: "<<elapsed.tv_sec*1000000000+elapsed.tv_nsec<<endl;

        // Test normalization
        for (int i = 0; i < num; ++i) {
            for (int k = 0; k < height; ++k) {
                for (int l = 0; l < width; ++l) {
                    // Test sum
                    Dtype sum = 0;
                    for (int j = 0; j < channels; ++j) {
                        sum += top_blob->data_at(i,j,k,l);
                    }
                    assert(sum >= 0.999);
                    assert(sum <= 1.001);

                    // Test exact values
                    Dtype scale = 0;
                    for (int j = 0; j < channels; ++j) {
                        scale += exp(bottom_blob->data_at(i,j,k,l));
                    }
                    for (int j = 0; j < channels; ++j) {
                        Dtype computed_value = top_blob->data_at(i,j,k,l);
                        Dtype expected_value = exp(bottom_blob->data_at(i,j,k,l))/scale;
                        //cout << computed_value << " " << expected_value << endl;
                        Dtype e = 1e-3;
                        assert(computed_value + e >= expected_value);
                        assert(computed_value - e <= expected_value);
                    }
                }
            }
        }
    }

private:
    int num = 2;
    int channels = 3;
    int height = 4;
    int width = 5;

    Blob<Dtype,Mtype>* bottom_blob;
    Blob<Dtype,Mtype>* top_blob;
    vector<Blob<Dtype,Mtype>*> bottom;
    vector<Blob<Dtype,Mtype>*> top;

    ofstream ofs;

    void init() {
        // Create bottom blob
        bottom_blob = new Blob<Dtype,Mtype>();
        bottom.push_back(bottom_blob);

        // Reshape bottom blob
        vector<int> shape {num, channels, height, width};
        bottom_blob->Reshape(shape);
        
        // Create top blob
        top_blob = new Blob<Dtype,Mtype>();
        top.push_back(top_blob);

        ofs.open("test_layer.log");
        clog.rdbuf(ofs.rdbuf());
    }

    string to_string(vector<int> shape) {
        string s = "";
        s += "(" + std::to_string(shape[0]);
        for (int i = 1; i < shape.size(); ++i) {
            s += "," + std::to_string(shape[i]);
        }
        s += ")";
        return s;
    }

    string to_string(Blob<Dtype,Mtype>* blob) {
        const Dtype* data = blob->cpu_data();
        int n = blob->num();
        int c = blob->channels();
        int h = blob->height();
        int w = blob->width();
        string s = "";
        for (int i = 0; i < n*c*h*w; ++i) {
            s += std::to_string(data[i]) + " ";
        }
        return s;
    }

    void init_rand(Blob<Dtype,Mtype>* blob) {
        Dtype* data = blob->mutable_cpu_data();
        for (int i = 0; i < blob->count(); ++i) {
            data[i] = Get<float16>(float(rand())/float(RAND_MAX));
        }
    }

    void init_ones(Blob<Dtype,Mtype>* blob) {
        Dtype* data = blob->mutable_cpu_data();
        srand(1234);
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
    //test.ConvolutionLayerTest();
    //test.InnerProductLayerTest();
    //test.PoolingLayerTest();
    //test.SplitLayerTest();
    //test.ReLULayerTest();
    test.SoftmaxLayerTest();
}
