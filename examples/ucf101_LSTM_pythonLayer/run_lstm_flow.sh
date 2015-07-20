#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/lisaanne/caffe-LSTM/examples/ucf101_LSTM_pythonLayer

#TO COME: determine which gpu has more memory

#for debugging python layer
GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_flow.prototxt -weights caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000  
#GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_flow.prototxt -weights caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000 > retrain_ucf101_flow_singleFrame.out 2>&1 
echo "Done."
