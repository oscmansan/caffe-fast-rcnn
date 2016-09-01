#!/bin/bash
PROGRAM=classification
if [ "$1" = "fp16" ]; then
    PROGRAM=classification_fp16
fi

#INPUT=../../examples/images/cat.jpg
#INPUT=../../../../inputs/ILSVRC2012_val_00000043.JPEG
#INPUT=../../../../inputs/ILSVRC2012_val_00047590.JPEG
#INPUT=../../../../inputs/ILSVRC2012_val_00041206.JPEG
#INPUT=../../../../inputs/ILSVRC2012_val_00027276.JPEG
#INPUT=../../../../inputs/ILSVRC2012_val_00014184.JPEG
#INPUT=../../../../inputs/ILSVRC2012_val_00016434.JPEG
#INPUT=../../../../inputs/ILSVRC2012_val_00028337.JPEG
#INPUT=../../../../inputs/ILSVRC2012_val_00038283.JPEG
INPUT=../../../../inputs/ILSVRC2012_val_00047965.JPEG

MODEL=bvlc_alexnet
#MODEL=bvlc_reference_caffenet
#MODEL=bvlc_googlenet

make $PROGRAM.bin

sudo LD_LIBRARY_PATH=/usr/local/cuda/lib:../../.build_release/lib \
./$PROGRAM.bin \
../../models/$MODEL/deploy.prototxt \
../../models/$MODEL/$MODEL.caffemodel \
../../data/ilsvrc12/imagenet_mean.binaryproto \
../../data/ilsvrc12/synset_words.txt \
$INPUT
