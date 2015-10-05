#!/usr/bin/env bash

DATA_DIR=../../examples/coco_caption/h5_data/
GPU_ID=1
WEIGHTS=\
/x/lisaanne/mrnn/snapshots_final/mrnn.direct_iter_110000.caffemodel,/x/lisaanne/coco_attribute/train_lexical_classifier/attributes_JJ100_NN300_VB100_zebra_iter_25000.caffemodel

#/home/lisaanne/caffe-LSTM/examples/coco_attribute/attributes_JJ100_NN300_VB100_iter_50000.caffemodel
../../build/tools/caffe train \
    -solver ../../examples/coco_attribute/attribute_mrnn_solver_fc8.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
