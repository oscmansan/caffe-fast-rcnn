#!/bin/bash
PROGRAM=test_layer

export CPLUS_INCLUDE_PATH=/usr/include/python2.7
export PYTHONPATH=/home/oscar/py-faster-rcnn-fp16/caffe-fast-rcnn/python:/home/oscar/py-faster-rcnn-fp16/lib

make clean
make $PROGRAM.bin
 
sudo LD_LIBRARY_PATH=/usr/local/cuda/lib:../../../../.build_release/lib ./$PROGRAM.bin

