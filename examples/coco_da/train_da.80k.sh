#!/usr/bin/env bash

GPU_ID=7
WEIGHTS=\
/x/lisaanne/pretrained_lm/mrnn.lm.direct_surf_lr0.01_iter_120000.caffemodel,attributes_JJ100_NN300_VB100_iter_50000.caffemodel
#/x/lisaanne/coco_caption/snapshots/coco_da_train_direct_attributes_ftPredictLM_pretrainAttributes_zebraClassifier_80k_iter_100000.solverstate
#/x/lisaanne/coco_caption/snapshots/coco_da_train_direct_attributes_lr0PredictLM_pretrainAttributes_zebraClassifier_80k_iter_80000.caffemodel
#/x/lisaanne/coco_caption/snapshots/coco_da_train_direct_attributes_lmPredict0_pretrainAttributes_zebra_iter_50000.caffemodel

DATA_DIR=../coco_caption/h5_data/

../../build/tools/caffe train \
    -solver da_solver.80k.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID

