#!/usr/bin/env bash

GPU_ID=1
WEIGHTS=\
/z/lisaanne/coco_caption/snapshots/coco_da_pretrainImageCap_FTPredictLM_zebraClassifier_80k_iter_110000.solverstate
#/z/lisaanne/coco_caption/snapshots/coco_da_pretrainImageCap_ACTUALLr0PredictLM_zebraClassifier_80k_iter_50000.caffemodel
#/z/lisaanne/coco_caption/snapshots/coco_da_pretrainImageCap_Lr0PredictLM_zebraClassifier_80k_iter_50000.solverstate
#/z/lisaanne/pretrained_lm/mrnn.lm.direct_imtext_lr0.01_cont60k_lr0.005_iter_100000.caffemodel,attributes_JJ100_NN300_VB100_zebra_iter_50000.caffemodel

DATA_DIR=../coco_caption/h5_data/

../../build/tools/caffe train \
    -solver da_solver.80k.ft.prototxt \
    -snapshot $WEIGHTS \
    -gpu $GPU_ID

