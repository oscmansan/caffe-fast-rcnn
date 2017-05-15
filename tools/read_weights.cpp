#include <string>
#include <iostream>
using namespace std;

#include "caffe/util/io.hpp"
using namespace caffe;

int main(int argc, char* argv[]) {
    NetParameter param;
    string file = argv[1];
    ReadProtoFromBinaryFile(file, &param);
    
    int num_source_layers = param.layer_size();
    for (int i = 0; i < num_source_layers; ++i) {
        const LayerParameter& source_layer = param.layer(i);
        const string& source_layer_name = source_layer.name();
	
	cout << source_layer_name << " ";

	for (int j = 0; j < source_layer.blobs_size(); ++j) {
	    BlobProto proto = source_layer.blobs(j);
	    vector<int> shape;
	    if (proto.has_num() || proto.has_channels() ||
        	proto.has_height() || proto.has_width()) {
            	// Using deprecated 4D Blob dimensions --
	    	// shape is (num, channels, height, width).
            	shape.resize(4);
      	    	shape[0] = proto.num();
     	    	shape[1] = proto.channels();
     	    	shape[2] = proto.height();
      	    	shape[3] = proto.width();
    	    } else {
      	    	shape.resize(proto.shape().dim_size());
      	    	for (int i = 0; i < proto.shape().dim_size(); ++i) {
        	    shape[i] = proto.shape().dim(i);
       	   	}
    	    }
	   
	    int count = shape[0]; 
	    for (int k = 1; k < shape.size(); ++k) {
		count *= shape[k];
	    }
	    cout << count << " ";
	    
	    if (source_layer.type() == "Mask") {
	        int N = shape[0];
	        int C = shape[1];
	        int H = shape[2];
	        int W = shape[3];
	        for (int i = 0; i < N; ++i) {
		    for (int j = 0; j < C; ++j) {
		        for (int k = 0; k < H; ++k) {
			    for (int l = 0; l < W; ++l) {
			        cout << proto.data(i*C*H*W+j*H*W+k*W+l) << " ";
			    }
			    cout << endl;
		        }
		        cout << endl;
		    }
	        }
	    }
	}
	cout << endl;
    }
}
