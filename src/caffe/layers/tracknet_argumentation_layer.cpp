
#include "caffe/layers/tracknet_argumentation_layer.hpp"

namespace caffe {

    template<typename Dtype>
    TrackNetArgumentationLayer<Dtype>::TrackNetArgumentationLayer(
       const LayerParameter &param) :
       Layer<Dtype>(param) {
       this->rng_ = boost::shared_ptr<cv::RNG>(new cv::RNG(0xFFFFFFFF));
       
       this->im_height_ = 224;
       this->im_width_ = 224;

       this->overseg_factor_ = 2;
       
       this->blur_lr_ = 1;
       this->blur_hr_ = 15;
       
    }

    template<typename Dtype>
    TrackNetArgumentationLayer<Dtype>::~TrackNetArgumentationLayer() {

    }

    template<typename Dtype>
    void TrackNetArgumentationLayer<Dtype>::LayerSetUp(
        const std::vector<caffe::Blob<Dtype>*>& bottom,
        const std::vector<caffe::Blob<Dtype>*>& top) {
       
    }

    template<typename Dtype>
    void TrackNetArgumentationLayer<Dtype>::Reshape(
       const std::vector<caffe::Blob<Dtype>* >& bottom,
       const std::vector<caffe::Blob<Dtype>* >& top) {
       //! only three channel images
       CHECK_EQ(bottom[0]->channels(), 3);
       //! # of labels and images are equal
       CHECK_EQ(bottom[0]->num(), bottom[1]->num());

       //! resize image output later
       //! note: top1 -> 224 x 224 template
       top[0]->Reshape(bottom[0]->num(),
                       bottom[0]->channels(),
                       im_height_,
                       im_width_);

       //! note: top1 -> 224 x 224 image
       top[1]->Reshape(bottom[0]->num(),
                       bottom[0]->channels(),
                       im_height_,
                       im_width_);

       top[2]->Reshape(bottom[1]->num(),
                       bottom[1]->channels(),
                       bottom[1]->height(),
                       bottom[1]->width());
    }

    template<typename Dtype>
    void TrackNetArgumentationLayer<Dtype>::Forward_cpu(
       const std::vector<caffe::Blob<Dtype> *>& bottom,
       const std::vector<caffe::Blob<Dtype> *>& top) {
       //! confirm size of image and labels
       CHECK_EQ(bottom[0]->num(), bottom[1]->num());

       /*
       const std::vector<Mat3v> input_images =
          blobToMats(*bottom[0]);
       const vector<vector<BboxLabel > > labels = blobToLabels(*bottom[1]);

       std::vector<Mat3v> output_images1(
          static_cast<int>(input_images.size()));
       std::vector<Mat3v> output_images2(
          static_cast<int>(input_images.size()));
       caffe::Blob<Dtype> &output_labels = *top[2];
       
       for (int i = 0; i < input_images.size(); i++) {
          const Mat3v *input_image = input_images[i];
          const std::vector<BboxLabel> &input_label = labels[i];
          
          Mat3v &output_image1 = output_images1[i];
          Mat3v &output_image2 = output_images2[i];
          Dtype *output_label = &output_labels.mutable_cpu_data()[
             output_labels.offset(i, 0, 0, 0)];

          Rectv in_rect;
          in_rect.x = input_label[i].bbox.x;
          in_rect.y = input_label[i].bbox.y;
          in_rect.width = input_label[i].bbox.width;
          in_rect.height = input_label[i].bbox.height;
          
          Rectv out_rect;
          transformImageCPU(output_image1, output_image2, out_rect,
                            input_image, in_rect);
          
          //! copy out_rect to blob <16, 17, 18, 19>
          output_labels[16] = out_rect.x;
          output_labels[17] = out_rect.y;
          output_labels[18] = out_rect.width;
          output_labels[19] = out_rect.height;
       }
       matsToBlob(output_images1, top[0]);
       matsToBlob(output_images2, top[1]);
       */
    }

    template<typename Dtype>
    void TrackNetArgumentationLayer<Dtype>::subtractImageMean(
       const Mat3v *in_image) {
       if (!this->mean_image_.empty()) {
          cv::subtract(*in_image, this->mean_image_, *in_image);
       }
    }

    template<typename Dtype>
    void TrackNetArgumentationLayer<Dtype>::transformImageCPU(
       Mat3v *templ_img, Mat3v *out_roi, Rectv *out_rect,
       const Mat3v *in_image, const Rectv *in_rect) {
       
       //! copy of inputs
       Mat3v image = in_image->clone();
       Rectv rect = *in_rect;

       //! subract mean if exist
       this->subtractImageMean(&image);

       this->shiftx_ = static_cast<Dtype>(this->rng_->uniform(-0.4, 0.4));
       this->shifty_ = static_cast<Dtype>(this->rng_->uniform(-0.4, 0.4));
       this->scale1_ = static_cast<Dtype>(this->rng_->uniform(0.7, 2.0));
       this->scale2_ = static_cast<Dtype>(this->rng_->uniform(0.9, 1.20));


       this->resizeImage(&image, &rect, this->scale1_, this->scale1_);

       //! adjust the bbox based on scale
       Pointv center = Pointv(in_rect->x + in_rect->width/2.0,
                              in_rect->y + in_rect->height/2.0);

       Rectv nrect;
       nrect.x = center.x - (in_rect->width * 0.5 * this->overseg_factor_);
       nrect.y = center.y - (in_rect->height * 0.5 * this->overseg_factor_);
       nrect.width = in_rect->width * this->overseg_factor_;
       nrect.height = in_rect->height * this->overseg_factor_;

       //! crop template image
       *templ_img = this->getSubwindow(image, center.x, center.y,
                                       nrect.width, nrect.height);
       
       Mat3v roi = randomTransformation(&nrect, &rect, &image, this->shiftx_,
                                         this->shifty_, this->scale2_);

       
       int wsize = static_cast<int>(this->rng_->uniform(this->blur_lr_,
                                                       this->blur_hr_));
       if (wsize % 2 && wsize > 2) {
          cv::GaussianBlur(roi, roi, cv::Size(wsize, wsize),
                           1, 1, cv::BORDER_REFLECT_101);
       }

       //! transform it to convnet input size
       transformToNetSize(&roi, &rect);
       transformToNetSize(templ_img, &nrect);

       *out_roi = roi.clone();
       *out_rect = rect;
    }

    template<typename Dtype>
    void TrackNetArgumentationLayer<Dtype>::transformToNetSize(
       Mat3v *image, Rectv *rect) {
       Dtype sx = static_cast<Dtype>(this->im_width_) /
          static_cast<Dtype>(image->cols);
       Dtype sy = static_cast<Dtype>(this->im_height_) /
          static_cast<Dtype>(image->rows);
       resizeImage(image, rect, sx, sy);
    }
    
    template<typename Dtype>
    void TrackNetArgumentationLayer<Dtype>::resizeImage(
       Mat3v *image, Rectv *rect, const Dtype scale_factor1,
       const Dtype scale_factor2) {
       if (scale_factor1 < 0.1f || scale_factor2 < 0.1f) {
          return;
       }
       float w = static_cast<Dtype>(image->cols) * scale_factor1;
       float h = static_cast<Dtype>(image->rows) * scale_factor2;
       cv::resize(*image, *image, cv::Size(static_cast<Dtype>(w),
                                           static_cast<Dtype>(h)));
       rect->x = static_cast<Dtype>(rect->x * scale_factor1);
       rect->y = static_cast<Dtype>(rect->y * scale_factor2);
       rect->width = static_cast<Dtype>(rect->width * scale_factor1);
       rect->height = static_cast<Dtype>(rect->height * scale_factor2);
    }


    template<typename Dtype>
    typename TrackNetArgumentationLayer<Dtype>::Mat3v
    TrackNetArgumentationLayer<Dtype>::randomTransformation(
       Rectv *rect, Rectv *bbox, const Mat3v *image, const Dtype sr_x,
       const Dtype sr_y, const Dtype scale) {
       Rectv new_box = *rect;

       new_box.x = rect->x + (sr_x * rect->width);
       new_box.y = rect->y + (sr_y * rect->height);
       
       //! shift
       cv::Rect new_rect = new_box;
       new_rect.x *= scale;
       new_rect.y *= scale;
       new_rect.width *= scale;
       new_rect.height *= scale;
       
       if (new_rect.x < 0) {
          new_rect.x = 0;
       }
       if (new_rect.x + new_rect.width > image->cols) {
          new_rect.x = new_rect.x - ((new_rect.x + new_rect.width) -
                                      image->cols);
       }
       if (new_rect.y < 0) {
          new_rect.y = 0;
       }
       if (new_rect.y + new_rect.height > image->rows) {
          new_rect.y = new_rect.y - ((new_rect.y + new_rect.height) -
                                       image->rows);
       }
       
       //! condition to avoid cropping the true object region
       if (new_rect.x > bbox->x) {
          new_rect.x = bbox->x;
       }
       if (new_rect.y > bbox->y) {
          new_rect.y = bbox->y;
       }
       if (new_rect.br().x < bbox->br().x) {
          new_rect.x += (bbox->br().x - new_rect.br().x);
       }
       if (new_rect.br().y < bbox->br().y) {
          new_rect.y += (bbox->br().y - new_rect.br().y);
       }
       
       //! check and correct boarders
       // if ((*bbox & new_rect).area() != bbox->area()) {
       //    // TODO(condition):  for overlapping boundary
       // }
       
       //! scale rect
       *rect = new_rect;
       Mat3v roi = (*image)(*rect).clone();
       
       //! update original box to current scale
       new_rect = Rectv();
       new_rect.x = bbox->x - rect->x;
       new_rect.y = bbox->y - rect->y;
       new_rect.width = bbox->width;
       new_rect.height = bbox->height;
       *bbox = new_rect;
       return roi;
       
    }

    template<typename Dtype>
    typename TrackNetArgumentationLayer<Dtype>::Mat3v
    TrackNetArgumentationLayer<Dtype>::getSubwindow(
       const Mat3v input, int cx, int cy, int width, int height) {
       Mat3v patch;
       int x1 = cx - width/2;
       int y1 = cy - height/2;
       int x2 = cx + width/2;
       int y2 = cy + height/2;

       if (x1 >= input.cols || y1 >= input.rows || x2 < 0 || y2 < 0) {
          patch = Mat3v(height, width, input.type());
          patch.setTo(0.f);
          return patch;
       }
       int top = 0, bottom = 0, left = 0, right = 0;
       if (x1 < 0) {
          left = -x1;
          x1 = 0;
       }
       if (y1 < 0) {
          top = -y1;
          y1 = 0;
       }
       if (x2 >= input.cols) {
          right = x2 - input.cols + width % 2;
          x2 = input.cols;
       } else {
          x2 += width % 2;
       }
       if (y2 >= input.rows) {
          bottom = y2 - input.rows + height % 2;
          y2 = input.rows;
       } else {
          y2 += height % 2;
       }
       if (x2 - x1 == 0 || y2 - y1 == 0) {
          patch = Mat3v(height, width, CV_32FC1);
          patch.setTo(0.f);
       } else {
          cv::copyMakeBorder(input(cv::Range(y1, y2),
                                   cv::Range(x1, x2)), patch,
                             top, bottom, left, right, cv::BORDER_REPLICATE);
       }
       assert(patch.cols == width && patch.rows == height);
       return patch;
    }

    template<typename Dtype>
    struct toDtype : public std::unary_function<float, Dtype> {
       Dtype operator() (const cv::Vec<Dtype, 1>& value) { return value(0); }
    };

    template<typename Dtype>
    void TrackNetArgumentationLayer<Dtype>::matToBlob(
       const Mat3v& source, Dtype* destination) const {
       std::vector<Mat1v> channels;
       cv::split(source, channels);
       size_t offset = 0;
       for (size_t iChannel = 0; iChannel != channels.size(); ++iChannel) {
          const Mat1v& channel = channels[iChannel];
          std::transform(
             channel.begin(),
             channel.end(),
             &destination[offset],
             toDtype<Dtype>());
          offset += channel.total();
       }
    }

    template<typename Dtype>
    void TrackNetArgumentationLayer<Dtype>::matsToBlob(
       const std::vector<Mat3v>& _source,
       Blob<Dtype>* _dest) const {
       for (size_t iImage = 0; iImage != _source.size(); ++iImage) {
         Dtype* destination = &_dest->mutable_cpu_data()[
            _dest->offset(iImage, 0, 0, 0)
            ];
         const Mat3v& source = _source[iImage];
         matToBlob(source, destination);
      }
    }
   
    template<typename Dtype>
    void TrackNetArgumentationLayer<Dtype>::Backward_cpu(
       const std::vector<caffe::Blob<Dtype> *>& bottom,
       const std::vector<bool>& propagate_down,
       const std::vector<caffe::Blob<Dtype> *>& top) {
       
    }
    
    INSTANTIATE_CLASS(TrackNetArgumentationLayer);
    REGISTER_LAYER_CLASS(TrackNetArgumentation);
   
}  // namespace caffe

