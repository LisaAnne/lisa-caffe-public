#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1

#for debugging python layer
GLOG_logtostderr=1  $TOOLS/caffe train -solver solver_attributes.prototxt -weights bvlc_reference_caffenet.caffemodel 
echo "Done."
