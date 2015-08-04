#!/usr/bin/env bash

GPU_ID=2
#WEIGHTS=snapshots/mrnn_attribute_JJ100_NN300_VB100_fc7_fc8-probs_iter_55000.solverstate
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

##########fc7
#WEIGHTS=snapshots/mrnn_attribute_JJ100_NN300_VB100_fc7_iter_55000.solverstate
#../../build/tools/caffe train \
#    -solver ../../examples/coco_attribute/attribute_mrnn_solver_fc7.prototxt \
#    -snapshot $WEIGHTS \
#    -gpu $GPU_ID
##########fc8
WEIGHTS=snapshots/attributes_JJ100_NN300_VB100_iter_50000.caffemodel
../../build/tools/caffe train \
    -solver ../../examples/coco_attribute/attribute_mrnn_solver_fc8.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID

