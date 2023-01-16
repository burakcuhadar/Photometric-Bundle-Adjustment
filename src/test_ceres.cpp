#include <iostream>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <opencv2/opencv2/imgproc.hpp>
#include <opencv2/opencv2/core/mat.hpp>
#include <pangolin/image/managed_image.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/image/image_io.h>

#include <visnav/common_types.h>
#include <tbb/concurrent_unordered_map.h>


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
    for(size_t i=0; i<img.w; i++) {
        for(size_t j=0; j<img.h; j++) {
            if(img_cv.at<uint8_t>(j,i) != img(i,j)) {
                std::cout << "error!" << std::endl;
            }
        }
    }

    std::cout <<"rows" << img_cv.rows << " cols " << img_cv.cols << std::endl;

    //const cv::Mat grad_col = grad_img.reshape(0, grad_img.rows*grad_img.cols); // new shape: [grad_img.h*grad_img.w, 1]
    const cv::Mat img_col = img_cv.reshape(0, img_cv.rows*img_cv.cols);
    uint8_t intensity1 = img_cv.at<uint8_t>(10,700);
    /*size_t x = (9*img_cv.cols + 700) % img_cv.cols;
    size_t y = (10*700) / img_cv.cols;
    uint8_t intensity2 = img_cv.at<uint8_t>(y, x);*/
    uint8_t intensity2 = img_col.at<uint8_t>(10*img_cv.cols + 700, 0);
    if(intensity1 != intensity2) {
        std::cout << "ERROR!!!" <<std::endl;
    }

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


