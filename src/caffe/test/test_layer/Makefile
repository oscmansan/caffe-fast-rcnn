BUILD_DIR := ../../../../build
CUDA_DIR := /usr/local/cuda

BUILD_INCLUDE_DIR := $(BUILD_DIR)/src
THIRDPARTY_DIR=$(PROJECT_DIR)/3rdparty
INCLUDE_DIRS += $(BUILD_INCLUDE_DIR) ../../../../src ../../../../include $(THIRDPARTY_DIR)
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
INCLUDE_DIRS += $(CUDA_INCLUDE_DIR)
CNMEM_DIR=${THIRDPARTY_DIR}/cnmem
INCLUDE_DIRS += ${CNMEM_DIR}/include
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
COMMON_FLAGS += -DUSE_CUDNN
COMMON_FLAGS += -DUSE_CNMEM
COMMON_FLAGS += -DUSE_OPENCV
WARNINGS := -Wall -Wno-sign-compare
WARNINGS += -Wno-uninitialized
LINKFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)

PROJECT := caffe
LIBRARY_NAME_SUFFIX := -nv
LIBRARY_NAME := $(PROJECT)$(LIBRARY_NAME_SUFFIX)

CUDA_LIB_DIR += $(CUDA_DIR)/lib
LIBRARY_DIRS += $(CUDA_LIB_DIR)
LIBRARY_DIRS += ${CNMEM_DIR}/build
LIB_BUILD_DIR := $(BUILD_DIR)/lib
LIBRARY_DIRS += $(LIB_BUILD_DIR)
LIBRARIES := cudart cublas curand
LIBRARIES += glog gflags protobuf boost_system m hdf5_hl hdf5
LIBRARIES += opencv_core opencv_highgui opencv_imgproc
LIBRARIES += boost_thread stdc++
LIBRARIES += cudnn
LIBRARIES += cnmem
LIBRARIES += cblas atlas
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) $(PKG_CONFIG) \
		$(foreach library,$(LIBRARIES),-l$(library))

ORIGIN := \$$ORIGIN

all: test_layer.bin

test_layer.bin: test_layer.cpp
	@ echo CXX/LD -o $@
	$(Q)g++ -std=c++11 -g $< -o $@ $(LINKFLAGS) -l$(LIBRARY_NAME) $(LDFLAGS) -Wl,-rpath,$(ORIGIN)/../../lib

clean:
	rm *.bin
