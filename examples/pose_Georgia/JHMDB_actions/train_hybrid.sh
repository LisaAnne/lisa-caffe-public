#!/bin/sh

export HDF5_DISABLE_VERSION_CHECK=1

#STEP 1
#../../build/tools/caffe train -solver ./solver_flow.prototxt -weights caffe_imagenet_train_iter_310000 > AZ_finetune_all_flow_SNAP2.out 2>&1
#/mnt/y/lisaanne/gdb-7.8/gdb/gdb --args ../../../build/tools/caffe train -solver solver_hybrid.prototxt -weights ucf101_RGB_single_frame_model.caffemodel 
../../../build/tools/caffe train -solver solver_hybrid.prototxt -weights ucf101_RGB_single_frame_model.caffemodel > JHMDB_single_frame_RGBJoints_debugInfo.out 2>&1
#../../build/tools/caffe train -solver solver_hybrid.prototxt -weights  caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000 > AZ_hybrid_RGB_finetuen_all_layers.out 2>&1

echo 'Done.'
