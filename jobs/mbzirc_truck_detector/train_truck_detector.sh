#!/usr/bin/env bash

cd $CAFFE_ROOT
build/tools/caffe train --solver=jobs/mbzirc_truck_detector/solver.prototxt \
    --gpu=0 \
    --weights=/home/krishneel/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel
    ##--weights=jobs/mbzirc_truck_detector/train3_weights/snapshot_iter_7800.caffemodel

