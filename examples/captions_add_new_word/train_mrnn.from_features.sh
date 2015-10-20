#!/usr/bin/env bash

DATA_DIR=../../examples/coco_caption/h5_data/

GPU_ID=1
#WEIGHTS=\
#/z/lisaanne/pretrained_lm/mrnn.direct_iter_110000.caffemodel
#,/z/lisaanne/lexical_models/attributes_JJ100_NN300_VB100_zebra_iter_50000.caffemodel

export PYTHONPATH=.

../../build/tools/caffe train \
    -solver ../../examples/captions_add_new_word/solver_mrnn_direct.from_features.prototxt \
    -gpu $GPU_ID
    #-weights $WEIGHTS \

