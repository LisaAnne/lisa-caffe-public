#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/lisaanne/caffe-forward-backward/examples/ucf101_LSTM_pythonLayer

#for debugging python layer
GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_flow.prototxt -weights single_frame_all_layers_hyb_flow_iter_70000.caffemodel > retrain_ucf101_flow_LSTM_resizeFix_lr01_flip_transpose.out 2>&1 
echo "Done."
