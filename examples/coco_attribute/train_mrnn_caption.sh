#!/usr/bin/env bash

GPU_ID=1
#WEIGHTS=snapshots/attributes_JJ100_NN300_VB100_iter_50000.caffemodel
WEIGHTS=snapshots/mrnn_attribute_JJ100_NN300_VB100_fc7_fc8-probs_iter_55000.solverstate
DATA_DIR=../../examples/coco_caption/h5_data/
if [ ! -d $DATA_DIR ]; then
    echo "Data directory not found: $DATA_DIR"
    echo "First, download the COCO dataset (follow instructions in data/coco)"
    echo "Then, run ./examples/coco_caption/coco_to_hdf5_data.py to create the Caffe input data"
    exit 1
fi

#/home/lisa/caffe-LSTM-video/build/tools/caffe train \
#    -solver /home/lisa/caffe-LSTM-video/examples/coco_caption/lrcn_solver_features.prototxt \
#    -weights $WEIGHTS \
#    -gpu $GPU_ID

../../build/tools/caffe train \
    -solver ../../examples/coco_attribute/attribute_mrnn_solver.prototxt \
    -snapshot $WEIGHTS \
    -gpu $GPU_ID
#    -snapshot /home/lisa/caffe-LSTM-video/examples/coco_caption/snapshots/lrcn_vgg_fromImages_iter_50000.solverstate \
#    -weights /home/lisa/caffe-LSTM-video/examples/extract_fc6_fc7/coco/VGG_ILSVRC_16_layers.caffemodel,/home/lisa/caffe-LSTM-video/examples/coco_caption/snapshots/lrcn_vgg_fromFeats_iter_110000.caffemodel \

