#!/usr/bin/env bash

/home/krishneel/nvcaffe/build/tools/caffe train --solver=/home/krishneel/nvcaffe/jobs/20161201-235008-141b/solver.prototxt \
    --gpu=0 \
    --weights=/home/krishneel/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel

