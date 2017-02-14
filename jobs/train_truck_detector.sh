#!/usr/bin/env bash
export CAFFE_ROOT=/home/krishneel/nvcaffe/
export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH
/home/krishneel/nvcaffe/build/tools/caffe train --solver=/home/krishneel/nvcaffe/jobs/20161201-235008-141b/solver.prototxt \
    --gpu=0 \
    --weights=/home/krishneel/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel

