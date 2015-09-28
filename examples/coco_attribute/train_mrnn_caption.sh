#!/usr/bin/env bash

DATA_DIR=../../examples/coco_caption/h5_data/
GPU_ID=0
WEIGHTS=\
/x/lisaanne/mrnn/snapshots_final/mrnn.direct_iter_110000.caffemodel,/home/lisaanne/caffe-LSTM/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
../../build/tools/caffe train \
    -solver ../../examples/coco_attribute/attribute_mrnn_solver_fc8.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
