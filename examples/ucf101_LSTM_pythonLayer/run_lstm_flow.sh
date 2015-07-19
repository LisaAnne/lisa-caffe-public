#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/lisaanne/caffe-forward-backward/examples/ucf101_LSTM_pythonLayer

#TO COME: determine which gpu has more memory

#GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver.prototxt -weights single_frame_all_layers_hyb_iter_5000.caffemodel 
GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_flow.prototxt -snapshot /mnt/y/lisaanne/snapshots_Jedits_lstm/snapshots_flow_LSTM_rerun_do0.9_iter_50000.solverstate > retrain_ucf101_flow_JulyTry_lr0.01_LSTMweights0.01_50k_to_110k.out 2>&1 
#GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_flow.prototxt -weights single_frame_all_layers_hyb_flow_iter_70000.caffemodel 
#GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver.prototxt -weights /mnt/y/lisaanne/snapshots_Jedits_lstm/snapshots_RGB_LSTM_rerun_iter_30000.caffemodel 
echo "Done."
