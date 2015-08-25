#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1

#for debugging python layer
#GLOG_logtostderr=1  $TOOLS/caffe train -solver solver_attributes.prototxt -weights snapshots/attributes_JJ100_NN300_VB100_iter_50000.caffemodel -gpu 2
GLOG_logtostderr=1 $TOOLS/caffe train -solver solver_attributes.prototxt -weights ../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 1 
echo "Done."
