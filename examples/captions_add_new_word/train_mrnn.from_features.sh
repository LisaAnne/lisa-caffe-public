#!/usr/bin/env bash

DATA_DIR=../../examples/coco_caption/h5_data/

GPU_ID=2
WEIGHTS=\
/x/lisaanne/mrnn/snapshots_final/mrnn.direct_iter_110000.caffemodel,/x/lisaanne/coco_attribute/train_lexical_classifier/attributes_JJ100_NN300_VB100_eightClusters_cocoImages_iter_50000.caffemodel

export PYTHONPATH=.

../../build/tools/caffe train \
    -solver ../../examples/captions_add_new_word/solver_mrnn_direct.from_features.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID

