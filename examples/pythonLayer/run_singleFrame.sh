#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1

#TO COME: determine which gpu has more memory

#GLOG_logtostderr=1  /mnt/y/lisaanne/gdb-7.8/gdb/gdb --args $TOOLS/caffe train -solver lstm_joints_solver.prototxt
GLOG_logtostderr=1  $TOOLS/caffe train -solver singleFrame_solver.prototxt -weights /home/lisaanne/caffe-forward-backward/examples/pose_Georgia/JHMDB_actions/ucf101_RGB_single_frame_model.caffemodel
# > testing_readImagePython_debugInfo.out 2>&1
#GLOG_logtostderr=1 $TOOLS/caffe train -solver lstm_joints_solver.prototxt #2>&1 run_joints_python_layer.out
#GLOG_logtostderr=1 $TOOLS/caffe train -solver lstm_joints_solver.prototxt 
echo "Done."
