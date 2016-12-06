#!/usr/bin/env bash

cd $CAFFE_ROOT
build/tools/caffe train --solver=jobs/mbzirc_truck_detector/solver.prototxt \
    --gpu=0 \
    --weights=jobs/mbzirc_truck_detector/weights/snapshot_iter_22801.caffemodel
## --weights=/home/krishneel/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel
