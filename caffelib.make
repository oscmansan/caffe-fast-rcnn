PROJECT := caffe

BUILD_DIR := build
BUILD_INCLUDE_DIR := $(BUILD_DIR)/src
CUDA_DIR := /usr/local/cuda

LIBRARY_NAME := $(PROJECT)
LIB_BUILD_DIR := $(BUILD_DIR)/lib
DYNAMIC_VERSION_MAJOR       := 1
DYNAMIC_VERSION_MINOR       := 0
DYNAMIC_VERSION_REVISION    := 0-rc3
DYNAMIC_NAME_SHORT := lib$(LIBRARY_NAME).so
DYNAMIC_VERSIONED_NAME_SHORT := $(DYNAMIC_NAME_SHORT).$(DYNAMIC_VERSION_MAJOR).$(DYNAMIC_VERSION_MINOR).$(DYNAMIC_VERSION_REVISION)
DYNAMIC_NAME := $(LIB_BUILD_DIR)/$(DYNAMIC_VERSIONED_NAME_SHORT)


# CXX_SRCS are the source files excluding the test ones.
CXX_SRCS := $(shell find src/$(PROJECT) ! -name "test_*.cpp" -name "*.cpp")
# CU_SRCS are the cuda source files
CU_SRCS := $(shell find src/$(PROJECT) ! -name "test_*.cu" -name "*.cu")
# PROTO_SRCS are the protocol buffer definitions
PROTO_SRC_DIR := src/$(PROJECT)/proto
PROTO_SRCS := $(wildcard $(PROTO_SRC_DIR)/*.proto)
PROTO_BUILD_DIR := $(BUILD_DIR)/$(PROTO_SRC_DIR)
PROTO_BUILD_INCLUDE_DIR := $(BUILD_INCLUDE_DIR)/$(PROJECT)/proto
PROTO_GEN_HEADER := $(addprefix $(PROTO_BUILD_INCLUDE_DIR)/, $(notdir ${PROTO_SRCS:.proto=.pb.h}))
PROTO_GEN_CC := $(addprefix $(BUILD_DIR)/, ${PROTO_SRCS:.proto=.pb.cc})
PY_PROTO_BUILD_DIR := python/$(PROJECT)/proto
PY_PROTO_INIT := python/$(PROJECT)/proto/__init__.py
PROTO_GEN_PY := $(foreach file,${PROTO_SRCS:.proto=_pb2.py}, \
        $(PY_PROTO_BUILD_DIR)/$(notdir $(file)))

# The objects corresponding to the source files
# These objects will be linked into the final shared library, so we
# exclude the tool, example, and test objects.
CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o})
CU_OBJS := $(addprefix $(BUILD_DIR)/cuda/, ${CU_SRCS:.cu=.o})
PROTO_OBJS := ${PROTO_GEN_CC:.cc=.o}
#OBJS := $(PROTO_OBJS) $(CXX_OBJS) $(CU_OBJS)


# Complete build flags.
THIRDPARTY_DIR=$(PROJECT_DIR)/3rdparty
INCLUDE_DIRS += $(BUILD_INCLUDE_DIR) src include $(THIRDPARTY_DIR)
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
INCLUDE_DIRS += $(CUDA_INCLUDE_DIR)
PYTHON_INCLUDE := /usr/include/python2.7 /usr/lib/python2.7/dist-packages/numpy/core/include
INCLUDE_DIRS += $(PYTHON_INCLUDE)
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
COMMON_FLAGS += -DUSE_CUDNN
COMMON_FLAGS += -DUSE_OPENCV
COMMON_FLAGS += -DWITH_PYTHON_LAYER
WARNINGS := -Wall -Wno-sign-compare
WARNINGS += -Wno-uninitialized
LINKFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)

CUDA_LIB_DIR += $(CUDA_DIR)/lib
LIBRARY_DIRS += $(CUDA_LIB_DIR)
LIB_BUILD_DIR := $(BUILD_DIR)/lib
LIBRARY_DIRS += $(LIB_BUILD_DIR)
PYTHON_LIB := /usr/lib
LIBRARY_DIRS += $(PYTHON_LIB)
LIBRARIES := cudart cublas curand
LIBRARIES += glog gflags protobuf boost_system m hdf5_hl hdf5
LIBRARIES += opencv_core opencv_highgui opencv_imgproc
LIBRARIES += boost_thread stdc++
LIBRARIES += cudnn
LIBRARIES += cblas atlas
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) $(PKG_CONFIG) \
            $(foreach library,$(LIBRARIES),-l$(library))

PYTHON_LIBRARIES ?= boost_python python2.7
PYTHON_LDFLAGS := $(LDFLAGS) $(foreach library,$(PYTHON_LIBRARIES),-l$(library))

NVCCFLAGS += -ccbin=g++ -Xcompiler -fPIC $(COMMON_FLAGS)

ORIGIN := $(CURDIR)


# PY$(PROJECT)_SRC is the python wrapper for $(PROJECT)
PY$(PROJECT)_SRC := python/$(PROJECT)/_$(PROJECT)_layer.cpp
PY$(PROJECT)_SO := python/$(PROJECT)/_$(PROJECT)_layer.so
PY$(PROJECT)_HXX := include/$(PROJECT)/layers/python_layer.hpp



FP16_CONVERSION := build/cuda/src/caffe/util/fp16_conversion.o
FLOAT16 := build/src/caffe/util/float16.o
CUDNN := build/src/caffe/util/cudnn.o
COMMON := build/src/caffe/common.o
CXX_MATH_FUNCTIONS := build/src/caffe/util/math_functions.o
CU_MATH_FUNCTIONS := build/cuda/src/caffe/util/math_functions.o
CXX_IM2COL := build/src/caffe/util/im2col.o
CU_IM2COL := build/cuda/src/caffe/util/im2col.o
SYNCEDMEM := build/src/caffe/syncedmem.o
BLOB := build/src/caffe/blob.o
LAYER := build/src/caffe/layer.o
BASE_CONV_LAYER := build/src/caffe/layers/base_conv_layer.o
CXX_CONV_LAYER := build/src/caffe/layers/conv_layer.o
CU_CONV_LAYER := build/cuda/src/caffe/layers/conv_layer.o
CXX_CUDNN_CONV_LAYER := build/src/caffe/layers/cudnn_conv_layer.o
CU_CUDNN_CONV_LAYER := build/cuda/src/caffe/layers/cudnn_conv_layer.o
CXX_INNER_PRODUCT_LAYER := build/src/caffe/layers/inner_product_layer.o
CU_INNER_PRODUCT_LAYER := build/cuda/src/caffe/layers/inner_product_layer.o
CXX_POOLING_LAYER := build/src/caffe/layers/pooling_layer.o
CU_POOLING_LAYER := build/cuda/src/caffe/layers/pooling_layer.o
CXX_CUDNN_POOLING_LAYER := build/src/caffe/layers/cudnn_pooling_layer.o
CU_CUDNN_POOLING_LAYER := build/cuda/src/caffe/layers/cudnn_pooling_layer.o
CXX_SPLIT_LAYER := build/src/caffe/layers/split_layer.o
CU_SPLIT_LAYER := build/cuda/src/caffe/layers/split_layer.o
NEURON_LAYER := build/src/caffe/layers/neuron_layer.o
CXX_RELU_LAYER := build/src/caffe/layers/relu_layer.o
CU_RELU_LAYER := build/cuda/src/caffe/layers/relu_layer.o
CXX_CUDNN_RELU_LAYER := build/src/caffe/layers/cudnn_relu_layer.o
CU_CUDNN_RELU_LAYER := build/cuda/src/caffe/layers/cudnn_relu_layer.o
CXX_SOFTMAX_LAYER := build/src/caffe/layers/softmax_layer.o
CU_SOFTMAX_LAYER := build/cuda/src/caffe/layers/softmax_layer.o
CXX_CUDNN_SOFTMAX_LAYER := build/src/caffe/layers/cudnn_softmax_layer.o
CU_CUDNN_SOFTMAX_LAYER := build/cuda/src/caffe/layers/cudnn_softmax_layer.o
CXX_RESHAPE_LAYER := build/src/caffe/layers/reshape_layer.o
CXX_ROI_POOLING_LAYER := build/src/caffe/layers/roi_pooling_layer.o
CU_ROI_POOLING_LAYER := build/cuda/src/caffe/layers/roi_pooling_layer.o
LAYER_FACTORY := build/src/caffe/layer_factory.o
NET := build/src/caffe/net.o
HDF5 := build/src/caffe/util/hdf5.o
INSERT_SPLITS := build/src/caffe/util/insert_splits.o
UPGRADE_PROTO := build/src/caffe/util/update_proto.o
IO := build/src/caffe/util/io.o

OBJS += $(PROTO_OBJS) $(CUDNN) $(FP16_CONVERSION) $(CXX_MATH_FUNCTIONS) $(CU_MATH_FUNCTIONS) $(CXX_IM2COL) $(CU_IM2COL) $(FLOAT16) $(COMMON) $(SYNCEDMEM) $(BLOB) $(LAYER) $(BASE_CONV_LAYER) $(CXX_CONV_LAYER) $(CU_CONV_LAYER) $(CXX_CUDNN_CONV_LAYER) $(CU_CUDNN_CONV_LAYER) $(CXX_INNER_PRODUCT_LAYER) $(CU_INNER_PRODUCT_LAYER) $(CXX_POOLING_LAYER) $(CU_POOLING_LAYER) $(CXX_CUDNN_POOLING_LAYER) $(CU_CUDNN_POOLING_LAYER) $(CXX_SPLIT_LAYER) $(CU_SPLIT_LAYER) $(NEURON_LAYER) $(CXX_RELU_LAYER) $(CU_RELU_LAYER) $(CXX_CUDNN_RELU_LAYER) $(CU_CUDNN_RELU_LAYER) $(CXX_SOFTMAX_LAYER) $(CU_SOFTMAX_LAYER) $(CXX_CUDNN_SOFTMAX_LAYER) $(CU_CUDNN_SOFTMAX_LAYER) $(CXX_RESHAPE_LAYER) $(CXX_ROI_POOLING_LAYER) $(CU_ROI_POOLING_LAYER) $(LAYER_FACTORY) $(NET) $(HDF5) $(INSERT_SPLITS) $(UPGRADE_PROTO) $(IO)


all: proto $(DYNAMIC_NAME) py

proto: $(PROTO_GEN_CC) $(PROTO_GEN_HEADER) $(PROTO_OBJS)

py: $(PY$(PROJECT)_SO) $(PROTO_GEN_PY)

$(PROTO_BUILD_DIR)/%.pb.cc $(PROTO_BUILD_DIR)/%.pb.h : $(PROTO_SRC_DIR)/%.proto | $(PROTO_BUILD_DIR)
	@ echo PROTOC $<
	@ protoc --proto_path=$(PROTO_SRC_DIR) --cpp_out=$(PROTO_BUILD_DIR) $<

$(PROTO_OBJS): $(PROTO_GEN_CC)
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(FP16_CONVERSION): src/caffe/util/fp16_conversion.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(FLOAT16): src/caffe/util/float16.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CUDNN): src/caffe/util/cudnn.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(COMMON): src/caffe/common.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CXX_MATH_FUNCTIONS): src/caffe/util/math_functions.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_MATH_FUNCTIONS): src/caffe/util/math_functions.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(CXX_IM2COL): src/caffe/util/im2col.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_IM2COL): src/caffe/util/im2col.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(SYNCEDMEM): src/caffe/syncedmem.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(BLOB): src/caffe/blob.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(LAYER): src/caffe/layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(BASE_CONV_LAYER): src/caffe/layers/base_conv_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CXX_CONV_LAYER): src/caffe/layers/conv_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_CONV_LAYER): src/caffe/layers/conv_layer.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(CXX_CUDNN_CONV_LAYER): src/caffe/layers/cudnn_conv_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_CUDNN_CONV_LAYER): src/caffe/layers/cudnn_conv_layer.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(CXX_INNER_PRODUCT_LAYER): src/caffe/layers/inner_product_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_INNER_PRODUCT_LAYER): src/caffe/layers/inner_product_layer.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(CXX_POOLING_LAYER): src/caffe/layers/pooling_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_POOLING_LAYER): src/caffe/layers/pooling_layer.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(CXX_CUDNN_POOLING_LAYER): src/caffe/layers/cudnn_pooling_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_CUDNN_POOLING_LAYER): src/caffe/layers/cudnn_pooling_layer.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(CXX_SPLIT_LAYER): src/caffe/layers/split_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(NEURON_LAYER): src/caffe/layers/neuron_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_SPLIT_LAYER): src/caffe/layers/split_layer.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(CXX_RELU_LAYER): src/caffe/layers/relu_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_RELU_LAYER): src/caffe/layers/relu_layer.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(CXX_CUDNN_RELU_LAYER): src/caffe/layers/cudnn_relu_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_CUDNN_RELU_LAYER): src/caffe/layers/cudnn_relu_layer.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(CXX_SOFTMAX_LAYER): src/caffe/layers/softmax_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_SOFTMAX_LAYER): src/caffe/layers/softmax_layer.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(CXX_CUDNN_SOFTMAX_LAYER): src/caffe/layers/cudnn_softmax_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_CUDNN_SOFTMAX_LAYER): src/caffe/layers/cudnn_softmax_layer.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(CXX_RESHAPE_LAYER): src/caffe/layers/reshape_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CXX_ROI_POOLING_LAYER): src/caffe/layers/roi_pooling_layer.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(CU_ROI_POOLING_LAYER): src/caffe/layers/roi_pooling_layer.cu
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(LAYER_FACTORY): src/caffe/layer_factory.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(NET): src/caffe/net.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(HDF5): src/caffe/util/hdf5.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(INSERT_SPLITS): src/caffe/util/insert_splits.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(UPGRADE_PROTO): src/caffe/util/upgrade_proto.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(IO): src/caffe/util/io.cpp
	@ echo CXX $<
	@ g++ $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS) -c $< -o $@

$(DYNAMIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	@ echo LD -o $@
	@ g++ -shared -o $@ $(OBJS) $(VERSIONFLAGS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS)
	@ cd $(BUILD_DIR)/lib; rm -f $(DYNAMIC_NAME_SHORT); ln -s $(DYNAMIC_VERSIONED_NAME_SHORT) $(DYNAMIC_NAME_SHORT)

$(PY$(PROJECT)_SO): $(PY$(PROJECT)_SRC) $(PY$(PROJECT)_HXX)
	@ echo CXX/LD -o $@ $<
	@ g++ -shared -o $@ $(PY$(PROJECT)_SRC) \
	    -o $@ $(LINKFLAGS) -l$(LIBRARY_NAME) $(PYTHON_LDFLAGS) \
        -Wl,-rpath,$(ORIGIN)/../../build/lib


clean:
	rm $(PROTO_GEN_CC) $(PROTO_OBJS) $(CUDNN) $(FP16_CONVERSION) $(CXX_MATH_FUNCTIONS) $(CU_MATH_FUNCTIONS) $(CXX_IM2COL) $(CU_IM2COL) $(FLOAT16) $(COMMON) $(SYNCEDMEM) $(BLOB) $(LAYER) $(BASE_CONV_LAYER) $(CXX_CONV_LAYER) $(CU_CONV_LAYER) $(CXX_CUDNN_CONV_LAYER) $(CU_CUDNN_CONV_LAYER) $(CXX_INNER_PRODUCT_LAYER) $(CU_INNER_PRODUCT_LAYER) $(CXX_POOLING_LAYER) $(CU_POOLING_LAYER) $(CXX_CUDNN_POOLING_LAYER) $(CU_CUDNN_POOLING_LAYER) $(CXX_SPLIT_LAYER) $(CU_SPLIT_LAYER) $(NEURON_LAYER) $(CXX_RELU_LAYER) $(CU_RELU_LAYER) $(CXX_CUDNN_RELU_LAYER) $(CU_CUDNN_RELU_LAYER) $(CXX_SOFTMAX_LAYER) $(CU_SOFTMAX_LAYER) $(CXX_CUDNN_SOFTMAX_LAYER) $(CU_CUDNN_SOFTMAX_LAYER) $(CXX_RESHAPE_LAYER) $(CXX_ROI_POOLING_LAYER) $(CU_ROI_POOLING_LAYER) $(LAYER_FACTORY) $(DYNAMIC_NAME) $(PY$(PROJECT)_SO) $(NET) $(HDF5) $(INSERT_SPLITS) $(UPGRADE_PROTO) $(IO)
