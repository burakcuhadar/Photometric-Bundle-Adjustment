#include <iostream>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <opencv2/opencv2/imgproc.hpp>
#include <opencv2/opencv2/core/mat.hpp>
#include <pangolin/image/managed_image.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/image/image_io.h>
#include <opencv2/opencv2/highgui.hpp>
#include <visnav/common_types.h>
#include <tbb/concurrent_unordered_map.h>
#include <opencv2/opencv2/imgcodecs.hpp>
#include <opencv2/opencv2/core.hpp>

using namespace visnav;

//double* img_ = new double[752*480];
//auto img_ = std::make_shared<double[]>(752*480);
//std::shared_ptr<double[]> img_(new double[752*480]());
std::vector<double> img_;

ceres::Grid2D<double, 1> img_to_grid(const pangolin::ManagedImage<uint8_t>& img) {
    const auto& h = img.h;
    const auto& w = img.w;

    //double img_[n];
    for(size_t j=0; j<h; j++) {

        for(size_t i=0; i<w; i++) {
            double val = (double) img(i, j);
            //img_[j*w+i] = val;
            img_.push_back(val);
        }
        //img_.push_back(double(img.ptr[i]));
    }
    std::cout << "created array" << std::endl;
    return ceres::Grid2D<double, 1>(img_.data(), 0, h, 0, w);
}//752 480

int main() {
    std::string dataset_path = "../data/euroc_V1/";
    pangolin::ManagedImage<uint8_t> img;

    std::stringstream ss;
    ss << dataset_path << "/1403715273262142976_0.jpg";

    pangolin::TypedImage img_ = pangolin::LoadImage(ss.str());
    img = std::move(img_);

    const cv::Mat img_cv(img.h, img.w, CV_8U, img.ptr);
    cv::Mat edges;
    cv::Canny(img_cv, edges, 100, 200);


    double min_inv_depth = 0.02;
    double max_inv_depth = 10;

    // clamp inv depth
    double clamped = 5;
    if(clamped < min_inv_depth) clamped = min_inv_depth;
    if(clamped > max_inv_depth) clamped = max_inv_depth;

    // Normalize inv_depth, range [0.02,10]
    int normed_inv_depth = (clamped - min_inv_depth) / (max_inv_depth - min_inv_depth) * 255;
    cv::Mat inv_depth_(1, 1, CV_8UC1);
    inv_depth_.at<int>(0,0) = normed_inv_depth;
    std::cout << inv_depth_.size() << std::endl;
    std::cout << inv_depth_.channels() << std::endl;
    //cv::Mat inv_depths[3] = {inv_depth_, inv_depth_.clone(),inv_depth_.clone()};
    //cv::Mat inv_depth3;
    //cv::merge(inv_depths, (size_t) 3, inv_depth3);
    cv::Mat color;
    cv::applyColorMap(inv_depth_, color, cv::COLORMAP_JET);
    //cv::cvtColor(inv_depth_, color, cv::COLOR_GRAY2RGB, 3);
    std::cout << color.size << std::endl;
    std::cout << color.channels() << std::endl;
    std::cout << color.type() << std::endl;
    std::cout << std::to_string(color.at<cv::Vec3b>(0,0)[0]) << std::endl;
    std::cout << std::to_string(color.at<cv::Vec3b>(0,0)[1]) << std::endl;
    std::cout << std::to_string(color.at<cv::Vec3b>(0,0)[2]) << std::endl;

    //cv::imshow("edges", edges);
    //cv::imshow("",img_cv);
    //cv::waitKey(0);

//    const cv::Mat img_cv(img.h, img.w, CV_8U, img.ptr);
//    for(size_t i=0; i<img.w; i++) {
//        for(size_t j=0; j<img.h; j++) {
//            if(img_cv.at<uint8_t>(j,i) != img(i,j)) {
//                std::cout << "error!" << std::endl;
//            }
//        }
//    }

//    std::cout <<"rows" << img_cv.rows << " cols " << img_cv.cols << std::endl;

    //const cv::Mat grad_col = grad_img.reshape(0, grad_img.rows*grad_img.cols); // new shape: [grad_img.h*grad_img.w, 1]
//    const cv::Mat img_col = img_cv.reshape(0, img_cv.rows*img_cv.cols);
//    uint8_t intensity1 = img_cv.at<uint8_t>(10,700);
    /*size_t x = (9*img_cv.cols + 700) % img_cv.cols;
    size_t y = (10*700) / img_cv.cols;
    uint8_t intensity2 = img_cv.at<uint8_t>(y, x);*/
//    uint8_t intensity2 = img_col.at<uint8_t>(10*img_cv.cols + 700, 0);
//    if(intensity1 != intensity2) {
//        std::cout << "ERROR!!!" <<std::endl;
//    }

    //const cv::Mat& block = img(cv::Rect(x_block * block_size, y_block * block_size, block_size, block_size));//TODO correct?
    //const cv::Mat& block = img_cv(cv::Rect(10,700,1, 1));
    //std::cout << (img_cv.at<uint8_t>(10,700) == block.at<uint8_t>(0,0)) << std::endl;
    /*for(size_t i=0; i<img.w; i++) {
        for(size_t j=0; j<img.h; j++) {
            if(img_cv.at<uint8_t>(j,i) != block.at<uint8_t>(j,i)) {
                std::cout << "error!" << std::endl;
                return 0;
            }
        }
    }*/


    /*ceres::Grid2D<double, 1> grid = img_to_grid(img);
    //ceres::Grid2D<uint8_t, 1> grid(img.begin(), 0, img.h, 0, img.w);

    double f;
    grid.GetValue(43,12,&f);
    std::cout << std::to_string(f) << std::endl;
    std::cout << std::to_string(img(12,43)) << std::endl;

    double ab[2] = {1.,2.};
    std::cout << ab[0] << " " << ab[1] << std::endl;

    //double f = 1e-5;
    //std::cout << std::to_string(f) << std::endl;*/

    return 0;
}


