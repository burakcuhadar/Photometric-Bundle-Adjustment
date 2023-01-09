#include <iostream>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

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

    ceres::Grid2D<double, 1> grid = img_to_grid(img);
    //ceres::Grid2D<uint8_t, 1> grid(img.begin(), 0, img.h, 0, img.w);

    double f;
    grid.GetValue(43,12,&f);
    std::cout << std::to_string(f) << std::endl;
    std::cout << std::to_string(img(12,43)) << std::endl;


    //double f = 1e-5;
    //std::cout << std::to_string(f) << std::endl;

    return 0;
}


