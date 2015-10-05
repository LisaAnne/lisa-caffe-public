#!/usr/bin/env bash

GPU_ID=1
WEIGHTS=/x/lisaanne/coco_attribute/train_lexical_classifier/attributes_JJ100_NN300_VB100_basicCap_lr0_iter_50000.caffemodel,/x/lisaanne/mrnn/snapshots_final/mrnn.direct_iter_110000.caffemodel

DATA_DIR=../coco_caption/h5_data/

../../build/tools/caffe train \
    -solver da_solver.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID

