#pragma once


#include <opengv/triangulation/methods.hpp>
#include <visnav/local_parameterization_se3.hpp>
#include <ceres/ceres.h>
#include <ceres/normal_prior.h>
#include <visnav/common_types.h>
#include <visnav/pba_types.h>
#include <visnav/photometric.h>
#include <thread>
#include <opencv2/opencv2/imgproc.hpp>
#include <opencv2/opencv2/core/mat.hpp>
#include <opencv2/opencv2/core.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <pangolin/image/managed_image.h>

namespace visnav {

//TODO more is needed?
const size_t region_size = 3;

struct PhotometricBundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;

  bool optimize_btf_params = true;
};

/*
ceres::Grid2D<double, 1> img_to_grid(const pangolin::ManagedImage<uint8_t>& img) {
    const auto& h = img.h;
    const auto& w = img.w;
    const size_t n = h * w;

    std::vector<double> img_;
    for(size_t i=0; i<n; i++) {
        img_.push_back(double(img.ptr[i]));
    }

    return ceres::Grid2D<double, 1>(img_.data(), 0, h, 0, w);
}*/


void photometric_bundle_adjustment(PbaLandmarks& landmarks,
                                   const std::set<FrameCamId>& fixed_cameras,
                                   Calibration& calib_cam,
                                   Cameras& cameras,
                                   const tbb::concurrent_unordered_map<FrameCamId, pangolin::ManagedImage<uint8_t>>& images,
                                   const std::map<FrameCamId, std::shared_ptr<ceres::Grid2D<double,1>>>& grids,
                                   const PhotometricBundleAdjustmentOptions& options) {
    ceres::Problem problem;

    const size_t& h = images.at({0,0}).h;
    const size_t& w = images.at({0,0}).w;

    // Setup optimization problem
    for (auto& cam : cameras) {
      problem.AddParameterBlock(cam.second.T_w_c.data(),
                                Sophus::SE3d::num_parameters,
                                new Sophus::test::LocalParameterizationSE3);
      if (fixed_cameras.count(cam.first) > 0) {
        std::cout << "fixing cam " << cam.first << std::endl;
        problem.SetParameterBlockConstant(cam.second.T_w_c.data());
      }

      problem.AddParameterBlock(cam.second.ab, 2);
      //TODO required?
      //problem.SetParameterLowerBound(&cam.second.a, 0, -5.0);
      //problem.SetParameterUpperBound(&cam.second.a, 0, 5.0);

      if (!options.optimize_btf_params) {
          problem.SetParameterBlockConstant(cam.second.ab);
          //problem.SetParameterBlockConstant(&cam.second.b);
      }

      if(*fixed_cameras.begin() == cam.first) {
          std::cout << "fixing btf parameters of " << cam.first << std::endl;
          problem.SetParameterBlockConstant(cam.second.ab);
          //problem.SetParameterBlockConstant(&cam.second.b);
      }
    }

    //TODO required?
    for(auto& landmark_kv : landmarks) {
        problem.AddParameterBlock(&landmark_kv.second.inv_depth, 1);
        //problem.SetParameterLowerBound(&landmark_kv.second.inv_depth, 0, 1.0/20.0);
        //problem.SetParameterUpperBound(&landmark_kv.second.inv_depth, 0, 1e8);
    }

    if (!options.optimize_intrinsics) {
      for (const auto& intrinsics : calib_cam.intrinsics) {
        problem.AddParameterBlock(intrinsics->data(), 8);
        problem.SetParameterBlockConstant(intrinsics->data());
      }
    }


    for(auto& landmark_kv : landmarks) {
        auto& lm = landmark_kv.second;

        // skip this landmark if inverse depth is an outlier //TODO does it make sense?
        if(lm.inv_depth < 0.0) {
            continue;
        }

        const auto& ref_fcid = lm.ref_frame;
        auto& ref_intrinsics = calib_cam.intrinsics[ref_fcid.cam_id];
        //const auto& ref_img = images.at(ref_fcid);

        // compute intensities of the pixels in the reference frame
        // also check whether all points of the pattern in the image, skip this landmark if not
        bool in_image = true;
        const auto ref_grid = *(grids.at(ref_fcid));
        ceres::BiCubicInterpolator<ceres::Grid2D<double,1>> ref_interp(ref_grid);
        std::vector<double> intensities;
        std::vector<Eigen::Vector2d> points;
        //std::vector<double> grad_norms;
        for(const auto& pattern : residual_pattern) {
            Eigen::Vector2d p(lm.p_2d(0) + pattern.first, lm.p_2d(1) + pattern.second);
            points.push_back(p);
            if(p(0) < 0 || p(0) > w || p(1) < 0 || p(1) > h) {
                in_image = false;
                break;
            }
            //TODO check also whether in target image?
            double intensity;
            //double dIdx;
            //double dIdy;
            ref_interp.Evaluate(p(1), p(0), &intensity);
            //intensities.push_back(ref_img(size_t(p(0)), size_t(p(1)))); //TODO this or with interp?
            intensities.push_back(intensity);
            //grad_norms.push_back(dIdx*dIdx + dIdy*dIdy);
        }
        if(!in_image) {
            continue;
        }

        // add residual blocks for every target frame
        for(const auto& target_fcid : lm.obs) {

            //const auto& target_fcid = feat_track_kv.first;
            if(target_fcid == ref_fcid) { //TODO normally lm.obs will not contain ref_fcid, remove this after correcting that!
                continue;
            }


            //const auto& target_img = images.at(target_fcid);
            const auto grid = *(grids.at(target_fcid));
            ceres::BiCubicInterpolator<ceres::Grid2D<double,1>> interp(grid);

            ceres::HuberLoss* huber = new ceres::HuberLoss(options.huber_parameter);
            PhotometricCostFunctor* cost_functor = new PhotometricCostFunctor(
                        points,
                        intensities,
                        ref_intrinsics->name(),
                        interp,
                        h,
                        w);

            if(target_fcid.cam_id == ref_fcid.cam_id) {
                ceres::CostFunction* cost_func = new ceres::AutoDiffCostFunction<PhotometricCostFunctor,
                        8, Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 8, 2, 2, 1>(cost_functor);

                problem.AddResidualBlock(
                            cost_func,
                            huber,
                            cameras.at(ref_fcid).T_w_c.data(),
                            cameras.at(target_fcid).T_w_c.data(),
                            calib_cam.intrinsics[ref_fcid.cam_id]->data(),
                            cameras.at(ref_fcid).ab,
                            cameras.at(target_fcid).ab,
                            &lm.inv_depth);
            }
            else {
                ceres::CostFunction* cost_func = new ceres::AutoDiffCostFunction<PhotometricCostFunctor,
                        8, Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 8, 8, 2, 2, 1>(cost_functor);

                problem.AddResidualBlock(
                            cost_func,
                            huber,
                            cameras.at(ref_fcid).T_w_c.data(),
                            cameras.at(target_fcid).T_w_c.data(),
                            calib_cam.intrinsics[ref_fcid.cam_id]->data(),
                            calib_cam.intrinsics[target_fcid.cam_id]->data(),
                            cameras.at(ref_fcid).ab,
                            cameras.at(target_fcid).ab,
                            &lm.inv_depth);
            }
        }
    }

    // Add normal priors for a&b
    /*for(auto& cam : cameras) {
        auto A = 0.001*Eigen::Matrix2d::Identity();
        auto b = Eigen::Vector2d::Zero();
        ceres::CostFunction* cost_func = new ceres::NormalPrior(A, b);
        problem.AddResidualBlock(cost_func, nullptr, cam.second.ab);
    }*/
    for(auto& cam: cameras) {
        BrightnessTransferRegularizer* reg_functor = new BrightnessTransferRegularizer();
        ceres::CostFunction* cost_func = new ceres::AutoDiffCostFunction<BrightnessTransferRegularizer, 2, 2>(reg_functor);
        problem.AddResidualBlock(cost_func, nullptr, cam.second.ab);
    }


    std::cout << "solving pba problem" << std::endl;
    // Solve
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = options.max_num_iterations;
    ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
    //ceres_options.preconditioner_type = ceres::SCHUR_JACOBI;//TODO required?
    ceres_options.num_threads = std::thread::hardware_concurrency();
    ceres::Solver::Summary summary;
    Solve(ceres_options, &problem, &summary);
    switch (options.verbosity_level) {
      // 0: silent
      case 1:
        std::cout << summary.BriefReport() << std::endl;
        break;
      case 2:
        std::cout << summary.FullReport() << std::endl;
        break;
    }
    std::cout << "pba problem solved!" << std::endl;

}

TrackId get_next_track_id(const PbaLandmarks& landmarks) {
    return landmarks.rbegin()->first + 1;
}

int tmp_count=100;

bool choose_pixel_from_block(const cv::Mat& grad_img, size_t& x, size_t& y) {

    const cv::Mat grad_col = grad_img.clone().reshape(0, grad_img.rows*grad_img.cols); // new shape: [grad_img.h*grad_img.w, 1]
    //std::cout << "row shape: " << row.size() << std::endl;
    // number of elements in the block(default value: 16*16=256)
    const int num = grad_col.rows;
    cv::Mat indices;
    cv::sortIdx(grad_col, indices, cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);
    //std::cout << "indices shape: " << indices.size() << std::endl;
    float median = 0.;
    if(num % 2 == 0) {
        int i_idx = indices.at<int>(num/2 - 1, 0);
        int j_idx = indices.at<int>(num/2, 0);
        float i = grad_col.at<float>(i_idx, 0);
        float j = grad_col.at<float>(j_idx, 0);
        median = (i+j)/2.;
    }
    else {
        int median_idx = indices.at<int>((num + 1) / 2 - 1, 0);
        median = grad_col.at<float>(median_idx, 0);
    }

    //TODO for debugging, remove
    /*if(tmp_count == 97) {
        std::cout << "grad col (rows,cols): " << grad_col.rows << "," << grad_col.cols << std::endl;
        std::cout << "indices (rows,cols): " << indices.rows << "," << indices.cols << std::endl;
        double min,max;
        cv::minMaxLoc(grad_col, &min, &max);
        std::cout << "min grad:" << min << " max grad: " << max << std::endl;
        std::cout << "indices min: " << grad_col.at<float>(indices.at<size_t>(0,0), 0);
        std::cout << "indices max: " << grad_col.at<float>(indices.at<size_t>(0,255), 0);
        std::cout << "grad row: " << grad_col << std::endl;
        std::cout << "indices: " << indices << std::endl;
    }*/

    // Choose the pixel with the highest gradient as a candidate point if it surpasses the threshold
    int candid_idx = indices.at<int>(0, 0);
    float candid_grad = grad_col.at<float>(candid_idx, 0);

    //TODO for debug, remove
    /*if(tmp_count > 0) {
        tmp_count--;
        std::cout << "median : " << median << " largest grad: " << candid_grad << std::endl;
    }*/


    if(candid_grad > (median + 500)) { //TODO in dso threshold was +7
        x = candid_idx % grad_img.cols;
        y = candid_idx / grad_img.cols;
        /*if(tmp_count > 0) {
            std::cout << "candid idx" << candid_idx << std::endl;
            std::cout << "cols " << grad_img.cols << std::endl;
            std::cout << "candid wrt block: " << std::to_string(x) << "," << std::to_string(y) << std::endl;
            tmp_count--;
        }*/
        return true;
    }

    return false;
}

bool region_out_of_bounds(const size_t& x,
                          const size_t& y,
                          const pangolin::ManagedImage<uint8_t>& img) {
    for(size_t i=0; i<region_size; i++) {
        for(size_t j=0; j<region_size; j++) {
            int curr_x = x - (region_size / 2) + i;
            int curr_y = y - (region_size / 2) + j;
            if(!img.InBounds(curr_x, curr_y)) {
                return true;
            }
        }
    }
    return false;
}

double compute_ncc(const cv::Point2i& ref_p,
                   const cv::Point2i& target_p,
                   const pangolin::ManagedImage<uint8_t>& ref_img,
                   const pangolin::ManagedImage<uint8_t>& target_img) {


    Eigen::Matrix<double, region_size, region_size> ref_region;
    Eigen::Matrix<double, region_size, region_size> target_region;
    for(size_t i=0; i<region_size; i++) {
        for(size_t j=0; j<region_size; j++) {
            ref_region(i,j) = ref_img(ref_p.x - (region_size / 2) + i, ref_p.y - (region_size / 2) + j);
            target_region(i,j) = target_img(target_p.x - (region_size / 2) + i, target_p.y - (region_size / 2) + j);
        }
    }

    double ref_mean = ref_region.mean();
    double target_mean = target_region.mean();

    double nominator = ((ref_region.array() - ref_mean) * (target_region.array() - target_mean)).sum();
    double denominator = sqrt((ref_region.array() - ref_mean).square().sum() *
                              (target_region.array() - target_mean).square().sum());

    return nominator / denominator;
}


void compute_best_match_along_epipolar_line(const size_t x,
                                            const size_t y,
                                            const Calibration& calib_cam,
                                            const FrameCamId& ref_fcid,
                                            const FrameCamId& target_fcid,
                                            const Sophus::SE3d& T_w_c1,
                                            const Sophus::SE3d& T_w_c2,
                                            const pangolin::ManagedImage<uint8_t>& ref_img,
                                            const pangolin::ManagedImage<uint8_t>& target_img,
                                            double& best_ncc,
                                            size_t& best_x,
                                            size_t& best_y) {
    //std::cout << "computing best match along epipolar line" << std::endl;
    const Eigen::Vector2d p((double) x, (double) y);
    const Sophus::SE3d T_c2_c1 = T_w_c2.inverse() * T_w_c1;

    Eigen::Vector3d unproj = calib_cam.intrinsics[ref_fcid.cam_id]->unproject(p).normalized();

    Eigen::Vector3d unproj_near = T_c2_c1 * (0.1 * unproj);
    Eigen::Vector3d unproj_far = T_c2_c1 * (50. * unproj);
    Eigen::Vector2d proj_near = calib_cam.intrinsics[target_fcid.cam_id]->project(unproj_near);
    Eigen::Vector2d proj_far = calib_cam.intrinsics[target_fcid.cam_id]->project(unproj_far);

    cv::Point2d p1(proj_near.x(), proj_near.y());
    cv::Point2d p2(proj_far.x(), proj_far.y());

    cv::Point2i p_int(p.x(), p.y());
    const cv::Mat target_img_cv(target_img.h, target_img.w, CV_8U, target_img.ptr);
    cv::LineIterator it(target_img_cv, p1, p2);
    best_ncc = -2.;
    cv::Point2i best_match;

    for(int i=0; i < it.count; i++, ++it) {
        const cv::Point2i target_p = it.pos();

        // Skip if target region out of bounds
        if(region_out_of_bounds(target_p.x, target_p.y, target_img)) {
            continue;
        }

        double ncc = compute_ncc(p_int, target_p, ref_img, target_img);
        if(ncc > best_ncc) {
            best_ncc = ncc;
            best_match = target_p;
        }
    }


    if(best_ncc > -2) {
        best_x = best_match.x;
        best_y = best_match.y;
    }
}

// for each sample point in each image, estimates an initial depth value and computes observations
// in other frames
void pba_landmarks_from_sample_pts(PbaLandmarks& landmarks,
                                   const std::vector<CandidatePoint> candids,
                                   const Cameras& cameras,
                                   const Calibration& calib_cam,
                                   const tbb::concurrent_unordered_map<FrameCamId, pangolin::ManagedImage<uint8_t>>& images) {

    for(const auto& candid : candids) {
        const auto& candid_fcid = candid.fcid;
        const auto& candid_x = candid.x;
        const auto& candid_y = candid.y;
        const auto& T_w_c_candid = cameras.at(candid_fcid).T_w_c;

        // Skip this candid if the region around the point is out of bounds
        if(region_out_of_bounds(candid_x, candid_y, images.at(candid_fcid))) {
            continue;
        }

        //TODO correct?
        double best_ncc = -2; // range of ncc is [-1,1]
        size_t best_x;
        size_t best_y;
        FrameCamId best_fcid;
        std::vector<FrameCamId> obs;

        //std::cout << "finding best match" << std::endl;
        // Go over every camera, find the best match along the epipolar line, estimate depth
        for(const auto& cam : cameras) {
            const auto& target_fcid = cam.first;
            if(target_fcid == candid_fcid) {
                continue;
            }

            const auto& T_w_c = cam.second.T_w_c;

            double ncc;
            size_t x;
            size_t y;
            compute_best_match_along_epipolar_line(candid_x,
                                                   candid_y,
                                                   calib_cam,
                                                   candid_fcid,
                                                   target_fcid,
                                                   T_w_c_candid,
                                                   T_w_c,
                                                   images.at(candid_fcid),
                                                   images.at(target_fcid),
                                                   ncc,
                                                   x,
                                                   y);

            if(ncc > best_ncc) {
                best_ncc = ncc;
                best_fcid = target_fcid;
                best_x = x;
                best_y = y;
            }
        }

        if(best_ncc < 0.9) {//TODO tweak?
            continue;
        }


        /*std::cout << "best ncc: " << best_ncc << std::endl;
        std::cout << "found best match" << std::endl;
        std::cout << "candid x and y: " << candid_x << "," << candid_y << std::endl;
        std::cout << "best x and y: " << best_x << "," << best_y << std::endl;*/

        // Estimate depth from best match
        using namespace opengv;
        const Sophus::SE3d& T_w_c0 = cameras.at(candid_fcid).T_w_c;
        const Sophus::SE3d& T_w_c1 = cameras.at(best_fcid).T_w_c;
        const Sophus::SE3d T_c0_c1 = T_w_c0.inverse() * T_w_c1;
        const Eigen::Matrix3d R01 = T_c0_c1.rotationMatrix();
        const Eigen::Vector3d t01 = T_c0_c1.translation();
        Eigen::Vector2d candid_point(candid_x, candid_y);
        Eigen::Vector2d target_point(best_x, best_y);
        bearingVectors_t bearingVectors0;
        bearingVectors_t bearingVectors1;
        bearingVectors0.push_back(calib_cam.intrinsics[candid_fcid.cam_id]->unproject(candid_point).normalized());
        bearingVectors1.push_back(calib_cam.intrinsics[best_fcid.cam_id]->unproject(target_point).normalized());
        relative_pose::CentralRelativeAdapter adapter(bearingVectors0,
                                                      bearingVectors1,
                                                      t01,
                                                      R01);
        Eigen::Vector3d p_c0 = triangulation::triangulate(adapter, 0);
        //std::cout << "triangulated" << std::endl;
        // if depth is outlier skip
        if(p_c0.norm() < 0.1 || p_c0.norm() > 30.) {
            continue;
        }
        double inv_depth = 1.0 / p_c0.norm();

        // Find the observations in other frames
        for(const auto& cam : cameras) {
            const auto& target_fcid = cam.first;
            if(target_fcid == candid_fcid) {
                continue;
            }
            const auto& target_img = images.at(target_fcid);
            const Sophus::SE3d& T_w_c = cam.second.T_w_c;
            const Eigen::Vector2d proj = calib_cam.intrinsics[target_fcid.cam_id]->project(T_w_c.inverse() * T_w_c0 * p_c0);
            if(target_img.InBounds(proj.x(), proj.y(), 0.)) {
                obs.push_back(target_fcid);
            }
        }

        if(obs.size() < 3){//TODO another num?
            continue;
        }

        // Create pba landmark
        TrackId tid = get_next_track_id(landmarks);
        //std::cout << "next track id: " << tid << std::endl;
        double intensity = images.at(candid_fcid)(candid_x, candid_y);
        //std::cout << "intensity: " << intensity << std::endl;
        PbaLandmark lm(candid_fcid, candid_point, inv_depth, intensity, obs);
        landmarks[tid] = lm;
    }

    //TODO remove sample point in a frame if it is observed before(how to organize the data structures for that?)
}


// Sample points similar to DSO (outliers need to be removed after calling this function)
void sample_points(PbaLandmarks& landmarks,
                   const Calibration& calib_cam,
                   const Cameras& cameras,
                   const tbb::concurrent_unordered_map<FrameCamId, pangolin::ManagedImage<uint8_t>>& images
                   ) {

    // Image size is 752x480, greatest common divisor is 16 so we look at 16x16 blocks
    const size_t w = images.begin()->second.w;
    const size_t h = images.begin()->second.h;
    const size_t block_size = 16;
    const size_t x_block_num = w / block_size;
    const size_t y_block_num = h / block_size;

    std::vector<CandidatePoint> candids;
    for(const auto& img_kv : images) {
        //std::cout << "Computing candids for image " << img_kv.first << std::endl;
        const auto& fcid = img_kv.first;
        const auto& img_raw = img_kv.second;
        const cv::Mat img(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);
        cv::Mat dx,dy;
        //TODO with cv::Canny() instead? as suggested by Sergei
        cv::Scharr(img, dx, CV_32F, 1, 0);
        cv::Scharr(img, dy, CV_32F, 0, 1);
        //cv::Mat grad_img = cv::abs(dx) + cv::abs(dy);
        cv::Mat grad_img;
        cv::sqrt(dx.mul(dx) + dy.mul(dy), grad_img);

        //TODO for debug, remove
        /*if(fcid == images.begin()->first) {
            cv::Mat dx, dy;
            cv::Scharr(img, dx, CV_32F, 1, 0);
            cv::Scharr(img, dy, CV_32F, 0, 1);
            cv::Mat grad_img = cv::abs(dx) + cv::abs(dy);
            double min,max;
            cv::minMaxLoc(grad_img, &min, &max);
            std::cout << "min grad:" << min << " max grad: " << max << std::endl;
        }*/

        for(size_t x_block=0; x_block < x_block_num; x_block++) {
            for(size_t y_block=0; y_block < y_block_num; y_block++) {
                //std::cout << "Computing candids for block:" << x_block << " " << y_block << std::endl;
                // For the current block compute the gradients
                const cv::Mat& block = grad_img(cv::Rect(x_block * block_size, y_block * block_size, block_size, block_size));
                //cv::Mat dx(block.size(), CV_32F);
                //cv::Mat dy(block.size(), CV_32F);
                //TODO with cv::Canny() instead? as suggested by Sergei
                //cv::Scharr(block, dx, CV_32F, 1, 0);
                //cv::Scharr(block, dy, CV_32F, 0, 1);
                //cv::Mat grad_img = cv::abs(dx) + cv::abs(dy);
                size_t candid_x;
                size_t candid_y;
                bool success = choose_pixel_from_block(block, candid_x, candid_y);
                // candid_x/y is wrt the patch, compute global coords
                if(success) {
                    candid_x += x_block * block_size;
                    candid_y += y_block * block_size;
                    candids.push_back({candid_x, candid_y, fcid});
                }
            }
        }
    }

    //TODO, debug, remove
    for(const auto& candid : candids) {
        if(!images.at(candid.fcid).InBounds(candid.x, candid.y, 0.)) {
            std::cout << "candid oob" << std::endl;
        }
    }

    std::cout << "candids size: " << candids.size() << std::endl;
    std::cout << "Creating pba landmarks from samples" << std::endl;
    std::cout << "landmark size before: " << landmarks.size() << std::endl;

    // Estimate depths for the selected candidate points, remove outliers
    pba_landmarks_from_sample_pts(landmarks,
                                  candids,
                                  cameras,
                                  calib_cam,
                                  images);
    std::cout << "Finished creating landmarks" << std::endl;
    std::cout << "landmark size after: " << landmarks.size() << std::endl;
}




/*
void photometric_bundle_adjustment(PbaLandmarks& landmarks,
                                   const std::set<FrameCamId>& fixed_cameras,
                                   Calibration& calib_cam,
                                   Cameras& cameras,
                                   const tbb::concurrent_unordered_map<FrameCamId, pangolin::ManagedImage<uint8_t>>& images,
                                   const PhotometricBundleAdjustmentOptions& options) {
    ceres::Problem problem;

    const size_t& h = images.at({0,0}).h;
    const size_t& w = images.at({0,0}).w;

    // Setup optimization problem
    for (auto& cam : cameras) {
      problem.AddParameterBlock(cam.second.T_w_c.data(),
                                Sophus::SE3d::num_parameters,
                                new Sophus::test::LocalParameterizationSE3);
      if (fixed_cameras.count(cam.first) > 0) {
        problem.SetParameterBlockConstant(cam.second.T_w_c.data());
      }

      if(*fixed_cameras.begin() == cam.first) {
          problem.AddParameterBlock(&cam.second.a, 1);
          problem.AddParameterBlock(&cam.second.b, 1);
          problem.SetParameterBlockConstant(&cam.second.a);
          problem.SetParameterBlockConstant(&cam.second.b);
      }
    }

    if (!options.optimize_intrinsics) {
      for (const auto& intrinsics : calib_cam.intrinsics) {
        problem.AddParameterBlock(intrinsics->data(), 8);
        problem.SetParameterBlockConstant(intrinsics->data());
      }
    }


    for(auto& landmark_kv : landmarks) {
        auto& lm = landmark_kv.second;

        // skip this landmark if inverse depth is an outlier //TODO does it make sense?
        if(lm.inv_depth < 1e-5 || lm.inv_depth > 15) {
            continue;
        }

        const auto& ref_fcid = lm.ref_frame;
        auto& ref_intrinsics = calib_cam.intrinsics[ref_fcid.cam_id];
        const auto& ref_img = images.at(ref_fcid);

        // compute intensities of the pixels in the reference frame
        // also check whether all points of the pattern in the image, skip this landmark if not
        bool in_image = true;
        ceres::Grid2D<uint8_t,1> ref_grid(ref_img.begin(), 0, h, 0, w);
        ceres::BiCubicInterpolator<ceres::Grid2D<uint8_t,1>> ref_interp(ref_grid);
        std::vector<uint8_t> intensities;
        for(const auto& pattern : residual_pattern) {
            Eigen::Vector2d p(lm.p_2d(0) + pattern.first, lm.p_2d(1) + pattern.second);
            if(p(0) < 0 || p(0) > w || p(1) < 0 || p(1) > h) {
                in_image = false;
                break;
            }
            double intensity;
            ref_interp.Evaluate(p(1), p(0), &intensity);
            intensities.push_back(intensity);
        }
        if(!in_image) {
            continue;
        }

        // add residual blocks for every target frame
        for(auto& feat_track_kv : lm.obs) {

            const auto& target_fcid = feat_track_kv.first;
            if(target_fcid == ref_fcid) { //TODO normally lm.obs will not contain ref_fcid, remove this after correcting that!
                continue;
            }


            const auto& target_img = images.at(target_fcid);
            ceres::Grid2D<uint8_t,1> grid(target_img.begin(), 0, target_img.h, 0, target_img.w);
            ceres::BiCubicInterpolator<ceres::Grid2D<uint8_t,1>> interp(grid);


            PhotometricCostFunctor* cost_functor = new PhotometricCostFunctor(
                        lm.p_2d,
                        intensities,
                        ref_intrinsics->name(),
                        interp,
                        target_img.h,
                        target_img.w);

            if(target_fcid.cam_id == ref_fcid.cam_id) {
                ceres::CostFunction* cost_func = new ceres::AutoDiffCostFunction<PhotometricCostFunctor,
                        pattern_size, Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 8, 1, 1, 1, 1, 1>(cost_functor);

                problem.AddResidualBlock(
                            cost_func,
                            new ceres::HuberLoss(options.huber_parameter),
                            cameras.at(ref_fcid).T_w_c.data(),
                            cameras.at(target_fcid).T_w_c.data(),
                            calib_cam.intrinsics[ref_fcid.cam_id]->data(),
                            &cameras.at(ref_fcid).a,
                            &cameras.at(ref_fcid).b,
                            &cameras.at(target_fcid).a,
                            &cameras.at(target_fcid).b,
                            &lm.inv_depth);
            }
            else {
                ceres::CostFunction* cost_func = new ceres::AutoDiffCostFunction<PhotometricCostFunctor,
                        pattern_size, Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 8, 8, 1, 1, 1, 1, 1>(cost_functor);

                problem.AddResidualBlock(
                            cost_func,
                            new ceres::HuberLoss(options.huber_parameter),
                            cameras.at(ref_fcid).T_w_c.data(),
                            cameras.at(target_fcid).T_w_c.data(),
                            calib_cam.intrinsics[ref_fcid.cam_id]->data(),
                            calib_cam.intrinsics[target_fcid.cam_id]->data(),
                            &cameras.at(ref_fcid).a,
                            &cameras.at(ref_fcid).b,
                            &cameras.at(target_fcid).a,
                            &cameras.at(target_fcid).b,
                            &lm.inv_depth);
            }
        }
    }

    std::cout << "solving pba problem" << std::endl;
    // Solve
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = options.max_num_iterations;
    ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres_options.num_threads = std::thread::hardware_concurrency();
    ceres::Solver::Summary summary;
    Solve(ceres_options, &problem, &summary);
    switch (options.verbosity_level) {
      // 0: silent
      case 1:
        std::cout << summary.BriefReport() << std::endl;
        break;
      case 2:
        std::cout << summary.FullReport() << std::endl;
        break;
    }
    std::cout << "pba problem solved!" << std::endl;
}
*/

}


