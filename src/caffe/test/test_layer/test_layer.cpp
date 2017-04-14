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
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/python_layer.hpp"
#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/get.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/cudnn_softmax_layer.hpp"
#endif
using namespace caffe;

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include <boost/foreach.hpp>
using boost::property_tree::ptree;

#define Dtype float16
#define Mtype float16

class LayerTest {
public:
    LayerTest() {
        Caffe::set_mode(Caffe::GPU);
        init();
    }

    void ConvolutionLayerTest() {
        cout << "## Testing ConvolutionLayer ############" << endl;

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
        cout << "## Testing InnerProductLayer ###########" << endl;

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
        cout << "## Testing PoolingLayer ################" << endl;

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
        cout << "## Testing SplitLayer ##################" << endl;

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

        // TODO: solve conflict with SoftmaxLayerTest
        top.clear();
        top_blob = new Blob<Dtype,Mtype>();
        top.push_back(top_blob);
    }

    void ReLULayerTest() {
        cout << "## Testing ReLULayer ###################" << endl;

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
        cout << "## Testing SoftmaxLayer ################" << endl;

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
                        // TODO: substract max to avoid numerical issues
                        scale += Get<Dtype>(exp(bottom_blob->data_at(i,j,k,l)));
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

    void TestFlattenOutputSizes() {
        // Set up layer parameters
        LayerParameter layer_param;
        BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
        blob_shape->add_dim(0);
        blob_shape->add_dim(-1),
        blob_shape->add_dim(1);
        blob_shape->add_dim(1);
       
        // Create layer
        ReshapeLayer<Dtype,Mtype> layer(layer_param);
        layer.SetUp(bottom,top);

        assert(top_blob->num() == num);
        assert(top_blob->channels() == channels * height * width);
        assert(top_blob->height() == 1);
        assert(top_blob->width() == 1);

        // Run forward pass
        timespec start,end,elapsed;
        clock_gettime(CLOCK_REALTIME,&start);
        layer.Forward(bottom,top);
        clock_gettime(CLOCK_REALTIME,&end);
        cout << "O: " << to_string(top_blob->shape()) << endl;
        clog << to_string(top_blob) << endl;
        elapsed = diff(start,end);
        cout<<"time: "<<elapsed.tv_sec*1000000000+elapsed.tv_nsec<<endl;

        // Test exact values
        for (int i = 0; i < num; ++i) {
            for (int j = 0; j < channels * height * width; ++j) {
                assert(top_blob->data_at(i, j, 0, 0) == bottom_blob->data_at(i, j / (height * width), (j / width) % height, j % width));
            }
        }
    }

    void TestInsertSingletonAxesStart() { 
        // Set up layer parameters
        LayerParameter layer_param;
        ReshapeParameter* reshape_param = layer_param.mutable_reshape_param();
        reshape_param->set_axis(0);
        reshape_param->set_num_axes(0);
        BlobShape* blob_shape = reshape_param->mutable_shape();
        blob_shape->add_dim(1),
        blob_shape->add_dim(1);
        blob_shape->add_dim(1);
       
        // Create layer
        ReshapeLayer<Dtype,Mtype> layer(layer_param);
        layer.SetUp(bottom,top);

        assert(top_blob->num_axes() == 7);
        assert(top_blob->shape(0) == 1);
        assert(top_blob->shape(1) == 1);
        assert(top_blob->shape(2) == 1);
        assert(top_blob->shape(3) == num);
        assert(top_blob->shape(4) == channels);
        assert(top_blob->shape(5) == height);
        assert(top_blob->shape(6) == width);
    }

    void TestInsertSingletonAxesMiddle() { 
        // Set up layer parameters
        LayerParameter layer_param;
        ReshapeParameter* reshape_param = layer_param.mutable_reshape_param();
        reshape_param->set_axis(2);
        reshape_param->set_num_axes(0);
        BlobShape* blob_shape = reshape_param->mutable_shape();
        blob_shape->add_dim(1),
        blob_shape->add_dim(1);
        blob_shape->add_dim(1);
       
        // Create layer
        ReshapeLayer<Dtype,Mtype> layer(layer_param);
        layer.SetUp(bottom,top);

        assert(top_blob->num_axes() == 7);
        assert(top_blob->shape(0) == num);
        assert(top_blob->shape(1) == channels);
        assert(top_blob->shape(2) == 1);
        assert(top_blob->shape(3) == 1);
        assert(top_blob->shape(4) == 1);
        assert(top_blob->shape(5) == height);
        assert(top_blob->shape(6) == width);
    }
    
    void TestInsertSingletonAxesEnd() { 
        // Set up layer parameters
        LayerParameter layer_param;
        ReshapeParameter* reshape_param = layer_param.mutable_reshape_param();
        reshape_param->set_axis(-1);
        reshape_param->set_num_axes(0);
        BlobShape* blob_shape = reshape_param->mutable_shape();
        blob_shape->add_dim(1),
        blob_shape->add_dim(1);
        blob_shape->add_dim(1);
       
        // Create layer
        ReshapeLayer<Dtype,Mtype> layer(layer_param);
        layer.SetUp(bottom,top);

        assert(top_blob->num_axes() == 7);
        assert(top_blob->shape(0) == num);
        assert(top_blob->shape(1) == channels);
        assert(top_blob->shape(2) == height);
        assert(top_blob->shape(3) == width);
        assert(top_blob->shape(4) == 1);
        assert(top_blob->shape(5) == 1);
        assert(top_blob->shape(6) == 1);
    }

    void ReshapeLayerTest() {
        cout << "## Testing ReshapeLayer ################" << endl;

        // Fill bottom blob
        init_rand(bottom_blob);
        cout << "I: " << to_string(bottom_blob->shape()) << endl; 
        clog << to_string(bottom_blob) << endl;

        TestFlattenOutputSizes();
        TestInsertSingletonAxesStart();
        TestInsertSingletonAxesMiddle();
        TestInsertSingletonAxesEnd();
    }

    void ROIPoolingLayerTest() {
        cout << "## Testing ROIPoolingLayer ################" << endl;

        // Fill bottom blob
        init_rand(bottom_blob);
        cout << "I: " << to_string(bottom_blob->shape()) << endl; 
        clog << to_string(bottom_blob) << endl;

        // Set up layer parameters
        LayerParameter layer_param;
        ROIPoolingParameter* roi_pooling_param = layer_param.mutable_roi_pooling_param();
        roi_pooling_param->set_pooled_h(1);
        roi_pooling_param->set_pooled_w(1);

        // Define ROIs
        Blob<Dtype,Mtype>* rois = new Blob<Dtype,Mtype>(4, 5, 1, 1);
        int i = 0;
        rois->mutable_cpu_data()[0 + 5*i] = 0;
        rois->mutable_cpu_data()[1 + 5*i] = 0;
        rois->mutable_cpu_data()[2 + 5*i] = 0;
        rois->mutable_cpu_data()[3 + 5*i] = 2;
        rois->mutable_cpu_data()[4 + 5*i] = 2;
        i = 1;
        rois->mutable_cpu_data()[0 + 5*i] = 0;
        rois->mutable_cpu_data()[1 + 5*i] = 2;
        rois->mutable_cpu_data()[2 + 5*i] = 0;
        rois->mutable_cpu_data()[3 + 5*i] = 4;
        rois->mutable_cpu_data()[4 + 5*i] = 2;
        i = 2;
        rois->mutable_cpu_data()[0 + 5*i] = 0;
        rois->mutable_cpu_data()[1 + 5*i] = 0;
        rois->mutable_cpu_data()[2 + 5*i] = 1;
        rois->mutable_cpu_data()[3 + 5*i] = 2;
        rois->mutable_cpu_data()[4 + 5*i] = 3;
        i = 3;
        rois->mutable_cpu_data()[0 + 5*i] = 0;
        rois->mutable_cpu_data()[1 + 5*i] = 2;
        rois->mutable_cpu_data()[2 + 5*i] = 1;
        rois->mutable_cpu_data()[3 + 5*i] = 4;
        rois->mutable_cpu_data()[4 + 5*i] = 3;
        bottom.push_back(rois);

        // Create layer
        ROIPoolingLayer<Dtype,Mtype> layer(layer_param);
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
    }

    void PythonLayerTest() {
        cout << "## Testing PythonLayer ################" << endl;

        // Create bottom blob
        int H = 38;  // height locations
        int W = 50;  // width locations
        int A = 9;   // anchors

        int shape[] = {1, 3};
        vector<int> im_info_shape(shape, shape+2);

        vector<Blob<Dtype,Mtype>*> bottom2;
        Blob<Dtype,Mtype>* scores = new Blob<Dtype,Mtype>(1,2*A,H,W);
        Blob<Dtype,Mtype>* bbox_deltas = new Blob<Dtype,Mtype>(1,4*A,H,W);
        Blob<Dtype,Mtype>* im_info = new Blob<Dtype,Mtype>(im_info_shape);
        bottom2.push_back(scores);
        bottom2.push_back(bbox_deltas);
        bottom2.push_back(im_info);

        // Fill bottom blob
        ifstream jsonFile("/home/oscar/bottom");
        ptree pt;
        read_json(jsonFile, pt);
        int i = 0;
        BOOST_FOREACH (ptree::value_type const& item, pt) {
            vector<float> vector_data = parse_blob<float>(item.second);
            Dtype* blob_data = bottom2[i]->mutable_cpu_data();
            assert(vector_data.size() == bottom2[i]->count());
            for (int j = 0; j < vector_data.size(); ++j) {
                blob_data[j] = Get<Dtype>(vector_data[j]);
            }
            ++i;
        }

        // Create top blob
        vector<Blob<Dtype,Mtype>*> top2;
        Blob<Dtype,Mtype>* rois = new Blob<Dtype,Mtype>();
        Blob<Dtype,Mtype>* scores2 = new Blob<Dtype,Mtype>();
        top2.push_back(rois);
        top2.push_back(scores2);

        // Set up layer parameters
        LayerParameter layer_param;
        layer_param.set_phase(TEST);
        layer_param.set_type("Python");
        PythonParameter* python_param = layer_param.mutable_python_param();
        python_param->set_module("rpn.proposal_layer");
        python_param->set_layer("ProposalLayer");
        python_param->set_param_str("'feat_stride': 16");

        // Create layer
        PythonLayer<float,float>* layer = (PythonLayer<float,float>*) LayerRegistry<float,float>::CreateLayer(layer_param).get();

        // Run forward pass
        timespec start,end,elapsed;
        try {
            layer->SetUp(bottom2,top2);
            clock_gettime(CLOCK_REALTIME,&start);
            layer->Forward(bottom2,top2);
            clock_gettime(CLOCK_REALTIME,&end);
        } 
        catch(bp::error_already_set) {
            PyErr_Print();
        }

        cout << "O: " << to_string(rois->shape()) << endl;
        cout << "O: " << to_string(scores2->shape()) << endl;
        clog << to_string(rois) << endl;
        clog << to_string(scores2) << endl;
        elapsed = diff(start,end);
        cout<<"time: "<<elapsed.tv_sec*1000000000+elapsed.tv_nsec<<endl;
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
        srand(1234);
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

    template <typename T>
    vector<T> parse_blob(ptree const& pt)
    {
        vector<T> data;
        BOOST_FOREACH (ptree::value_type const& item, pt) {
            try {
                data.push_back(item.second.get_value<T>());
            }
            catch(exception const& e) {
                vector<T> data2 = parse_blob<T>(item.second);
                BOOST_FOREACH (T x, data2) {
                    data.push_back(x);
                }
            }
        }
        return data;
    }
};


int main() {
    LayerTest test;
    //test.ConvolutionLayerTest();
    //test.InnerProductLayerTest();
    //test.PoolingLayerTest();
    //test.SplitLayerTest();
    //test.ReLULayerTest();
    //test.SoftmaxLayerTest();
    //test.ReshapeLayerTest();
    //test.ROIPoolingLayerTest();
    test.PythonLayerTest();
}
