#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    for (int i = 0; i < bottom.size(); i++) {
       Dtype* top_data1 = bottom[i]->mutable_cpu_data();
       std::cout << bottom.size()  << "\t" << bottom[i]->num()<< "\n";
       std::cout << bottom.size()  << "\t" << bottom[i]->height()<< "\n";
       std::cout << bottom.size()  << "\t" << bottom[i]->width()<< "\n";
       std::cout << bottom.size()  << "\t" << bottom[i]->channels()<< "\n";

       std::cout << "PRINTING " << i  << " " << "\n";
       int index = -1;
       for (int k = 0; k < bottom[i]->channels(); k++) {
          for (int y = 0; y < bottom[i]->height(); y++) {
             int t = 0;
             for (int x = 0; x < bottom[i]->width(); x++) {
                index = y + (x * bottom[i]->height()) +
                   (k * bottom[i]->width() * bottom[i]->height());
                std::cout << top_data1[index] << " ";
                t++;
             }
             std::cout  << "\n";
          }
          std::cout << "\nNext Channel:  " << index  << "\n";
       }
       std::cout  << "\n\n";
    }
   
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
