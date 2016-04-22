#!/usr/bin/env bash

GPU_ID=2
WEIGHTS=\
snapshots/lrcn_vgg_fromImages_rm1_lr0p01_iter_90000.caffemodel
#snapshots/lrcn_vgg_fromImages_rm1_iter_90000.caffemodel
DATA_DIR=h5_data/
if [ ! -d $DATA_DIR ]; then
    echo "Data directory not found: $DATA_DIR"
    echo "First, download the COCO dataset (follow instructions in data/coco)"
    echo "Then, run ./examples/coco_caption/coco_to_hdf5_data.py to create the Caffe input data"
    exit 1
fi

../../build/tools/caffe train \
    -solver lrcn_finetune_solver.vgg.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
