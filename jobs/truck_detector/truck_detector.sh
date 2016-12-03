cd /home/krishneel/caffe
./build/tools/caffe train \
    --solver="jobs/truck_detector/solver.prototxt" \
    --weights="models/bvlc_googlenet/bvlc_googlenet.caffemodel" \
    --gpu 0 2>&1 | tee jobs/truck_detector/truck_detector.log
