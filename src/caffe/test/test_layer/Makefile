BUILD_DIR := ../../../../build
CUDA_DIR := /usr/local/cuda

BUILD_INCLUDE_DIR := $(BUILD_DIR)/src
THIRDPARTY_DIR=$(PROJECT_DIR)/3rdparty
INCLUDE_DIRS += $(BUILD_INCLUDE_DIR) ../../../../src ../../../../include $(THIRDPARTY_DIR)
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
INCLUDE_DIRS += $(CUDA_INCLUDE_DIR)
INCLUDE_DIRS += ${CNMEM_DIR}/include
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
COMMON_FLAGS += -DUSE_CUDNN
COMMON_FLAGS += -DUSE_OPENCV
WARNINGS := -Wall -Wno-sign-compare
WARNINGS += -Wno-uninitialized
LINKFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)

PROJECT := caffe
LIBRARY_NAME := $(PROJECT)

CUDA_LIB_DIR += $(CUDA_DIR)/lib
LIBRARY_DIRS += $(CUDA_LIB_DIR)
LIB_BUILD_DIR := $(BUILD_DIR)/lib
LIBRARY_DIRS += $(LIB_BUILD_DIR)
LIBRARIES := cudart cublas curand
LIBRARIES += glog gflags protobuf boost_system m hdf5_hl hdf5
LIBRARIES += opencv_core opencv_highgui opencv_imgproc
LIBRARIES += boost_thread stdc++
LIBRARIES += cudnn
LIBRARIES += cblas atlas
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) $(PKG_CONFIG) \
		$(foreach library,$(LIBRARIES),-l$(library))

OBJS += /src/caffe/proto/caffe.pb.o
OBJS += /src/caffe/util/float16.o
OBJS += /src/caffe/util/cudnn.o
OBJS += /cuda/src/caffe/util/fp16_conversion.o
OBJS += /src/caffe/syncedmem.o
OBJS += /src/caffe/common.o
OBJS += /src/caffe/util/math_functions.o
OBJS += /cuda/src/caffe/util/math_functions.o
OBJS += /src/caffe/util/im2col.o
OBJS += /cuda/src/caffe/util/im2col.o
OBJS += /src/caffe/blob.o
OBJS += /src/caffe/layer.o
OBJS += /src/caffe/layers/base_conv_layer.o
OBJS += /src/caffe/layers/conv_layer.o
OBJS += /cuda/src/caffe/layers/conv_layer.o
OBJS += /src/caffe/layers/inner_product_layer.o
OBJS += /cuda/src/caffe/layers/inner_product_layer.o
OBJS += /src/caffe/layers/pooling_layer.o
OBJS += /cuda/src/caffe/layers/pooling_layer.o
OBJS += /src/caffe/layers/split_layer.o
OBJS += /cuda/src/caffe/layers/split_layer.o
OBJS += /src/caffe/layers/neuron_layer.o
OBJS += /src/caffe/layers/relu_layer.o
OBJS += /cuda/src/caffe/layers/relu_layer.o
OBJS += /src/caffe/layers/softmax_layer.o
OBJS += /cuda/src/caffe/layers/softmax_layer.o
OBJS += /src/caffe/layers/cudnn_softmax_layer.o
OBJS += /cuda/src/caffe/layers/cudnn_softmax_layer.o
OBJS += /src/caffe/layers/reshape_layer.o
OBJS += /src/caffe/layers/roi_pooling_layer.o
OBJS += /cuda/src/caffe/layers/roi_pooling_layer.o
OBJS += /src/caffe/layer_factory.o
OBJS := $(foreach obj, $(OBJS), $(BUILD_DIR)$(obj))

ORIGIN := \$$ORIGIN

all: test_layer.bin

test_layer.bin: test_layer.cpp
	@ echo CXX/LD -o $@
	@ g++ -std=c++11 -g $< -o $@ $(OBJS) $(LINKFLAGS) $(LDFLAGS) -Wl,-rpath,$(ORIGIN)/../../lib

clean:
	rm *.bin
