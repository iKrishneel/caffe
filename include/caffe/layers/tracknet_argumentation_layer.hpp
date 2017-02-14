
// #pragma once
#ifndef TRACKNET_ARGUMENTATION_LAYER_HPP
#define TRACKNET_ARGUMENTATION_LAYER_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <boost/array.hpp>
#include <boost/foreach.hpp>
#include <boost/static_assert.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
// #include "caffe/layers/detectnet_transform_layer.hpp"

namespace caffe {

template<typename Dtype>
struct BboxLabel_;
   
template<typename Dtype>
class TrackNetArgumentationLayer: public Layer<Dtype> {

 public:
    explicit TrackNetArgumentationLayer(const LayerParameter& param);
    virtual ~TrackNetArgumentationLayer();
    virtual void LayerSetUp(const std::vector<caffe::Blob<Dtype>*>& bottom,
                            const std::vector<caffe::Blob<Dtype>*>& top);
    virtual void Reshape(const std::vector<caffe::Blob<Dtype>* >& bottom,
                         const std::vector<caffe::Blob<Dtype>* >& top);
   
    virtual inline const char* type() const {return "TrackNetArgumentation"; }
    virtual inline int ExactNumBottomBlobs() const {return 2; }
    virtual inline int ExactNumTopBlobs() const {return 3; }

 protected:
    virtual void Forward_cpu(
       const std::vector<caffe::Blob<Dtype> *>& bottom,
       const std::vector<caffe::Blob<Dtype> *>& top);
    virtual void Backward_cpu(
       const std::vector<caffe::Blob<Dtype> *>& bottom,
       const std::vector<bool>& propagate_down,
       const std::vector<caffe::Blob<Dtype> *>& top);

    boost::shared_ptr<cv::RNG> rng_;

   
 private:
    typedef BboxLabel_<Dtype> BboxLabel;
    typedef cv::Mat_<cv::Vec<Dtype, 3> > Mat3v;
    typedef cv::Mat_<cv::Vec<Dtype, 1> > Mat1v;
    typedef cv::Rect_<Dtype> Rectv;
    typedef cv::Point_<Dtype> Pointv;
   
    int im_height_;  //! change to proto
    int im_width_;
    Dtype overseg_factor_;

    //! scale image
    Dtype scale1_;
    Dtype scale2_;
   
    //! shifting ratio
    Dtype shiftx_;
    Dtype shifty_;

    //! blur window
    int blur_lr_;
    int blur_hr_;

    Mat3v mean_image_;
    virtual void subtractImageMean(
       const Mat3v *in_image);
    virtual void transformImageCPU(
       Mat3v *templ_img, Mat3v * out_roi, Rectv *out_rect,
       const Mat3v *in_image, const Rectv *in_rect);
    virtual void resizeImage(
      Mat3v *image, Rectv *rect, const Dtype scale_factor1,
      const Dtype scale_factor2);
    virtual Mat3v getSubwindow(
       const Mat3v input, int cx, int cy, int width, int height);

    virtual Mat3v randomTransformation(
       Rectv *rect, Rectv *bbox, const Mat3v *image, const Dtype sr_x,
       const Dtype sr_y, const Dtype scale);
    virtual void transformToNetSize(
       Mat3v *image, Rectv *rect);

    virtual void matToBlob(
       const Mat3v& source, Dtype* destination) const;
    virtual void matsToBlob(
       const std::vector<Mat3v>& source, Blob<Dtype>* destination) const;
    // virtual std::vector<Mat3v> blobToMats(
    //   const Blob<Dtype>& image) const;
    // virtual std::vector<std::vector<BboxLabel> > blobToLabels(
    //   const Blob<Dtype>& labels) const;
   
   
   
};



}  // namespace caffe

#endif /* TRACKNET_ARGUMENTATION_LAYER_HPP */
