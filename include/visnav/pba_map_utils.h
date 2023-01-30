#pragma once

#include <fstream>
#include <algorithm>
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
#include <visnav/serialization.h>


namespace visnav {


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

      if (!options.optimize_btf_params) {
          problem.SetParameterBlockConstant(cam.second.ab);
      }

      if(*fixed_cameras.begin() == cam.first) {
          std::cout << "fixing btf parameters of " << cam.first << std::endl;
          problem.SetParameterBlockConstant(cam.second.ab);
      }
    }

    for(auto& landmark_kv : landmarks) {
        problem.AddParameterBlock(&landmark_kv.second.inv_depth, 1);
    }

    if (!options.optimize_intrinsics) {
      for (const auto& intrinsics : calib_cam.intrinsics) {
        problem.AddParameterBlock(intrinsics->data(), 8);
        problem.SetParameterBlockConstant(intrinsics->data());
      }
    }


    for(auto& landmark_kv : landmarks) {
        auto& lm = landmark_kv.second;

        // skip this landmark if inverse depth is an outlier
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
            double intensity;
            //double dIdx;
            //double dIdy;
            ref_interp.Evaluate(p(1), p(0), &intensity);
            //intensities.push_back(ref_img(size_t(p(0)), size_t(p(1))));
            intensities.push_back(intensity);
            //grad_norms.push_back(dIdx*dIdx + dIdy*dIdy);
        }
        if(!in_image) {
            continue;
        }

        // add residual blocks for every target frame
        for(const auto& target_fcid : lm.obs) {

            //const auto& target_fcid = feat_track_kv.first;
            if(target_fcid == ref_fcid) {
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
                        pattern_size, Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 8, 2, 2, 1>(cost_functor);

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
                        pattern_size, Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 8, 8, 2, 2, 1>(cost_functor);

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
    //ceres_options.preconditioner_type = ceres::SCHUR_JACOBI;
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

bool residual_pattern_out_of_bounds(const size_t& x,
                                    const size_t& y,
                                    const pangolin::ManagedImage<uint8_t>& img) {
    for(const auto& pattern : residual_pattern) {
        size_t pattern_x = x + pattern.first;
        size_t pattern_y = y + pattern.second;
        if(!img.InBounds((int) pattern_x, (int) pattern_y)) {
            return true;
        }
    }
    return false;
}

/*
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

    Eigen::Vector3d unproj = calib_cam.intrinsics[ref_fcid.cam_id]->unproject(p);

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
*/

/*
void compute_residual(const size_t& ref_x,
                      const size_t& ref_y,
                      const size_t& target_x,
                      const size_t& target_y,
                      const Calibration& calib_cam,
                      const FrameCamId& ref_fcid,
                      const FrameCamId& target_fcid,
                      const Sophus::SE3d& T_c1_c0,
                      //const Sophus::SE3d& T_w_c0,
                      //const Sophus::SE3d& T_w_c1,
                      const double* ref_ab,
                      const double* target_ab,
                      const pangolin::ManagedImage<uint8_t>& ref_img,
                      const pangolin::ManagedImage<uint8_t>& target_img,
                      double& residual,
                      double& inv_depth,
                      size_t& proj_x,
                      size_t& proj_y) {

    // Estimate depth
    using namespace opengv;
    //const Sophus::SE3d T_c1_c0 = T_w_c1.inverse() * T_w_c0;
    const Sophus::SE3d T_c0_c1 = T_c1_c0.inverse();
    const Eigen::Matrix3d R01 = T_c0_c1.rotationMatrix();
    const Eigen::Vector3d t01 = T_c0_c1.translation();
    Eigen::Vector2d target_point(target_x, target_y);
    Eigen::Vector2d ref_point(ref_x, ref_y);
    bearingVectors_t bearingVectors0;
    bearingVectors_t bearingVectors1;
    bearingVectors0.push_back(calib_cam.intrinsics[ref_fcid.cam_id]->unproject(ref_point).stableNormalized());
    bearingVectors1.push_back(calib_cam.intrinsics[target_fcid.cam_id]->unproject(target_point).stableNormalized());
    relative_pose::CentralRelativeAdapter adapter(bearingVectors0,
                                                  bearingVectors1,
                                                  t01,
                                                  R01);
    Eigen::Vector3d p_c0 = triangulation::triangulate(adapter, 0);
    // if depth is too small, skip this
    if(p_c0.norm() < 0.1 || p_c0.z() < 0.05) {
        residual = std::numeric_limits<double>::max();
        inv_depth = -1;
        return;
    }
    inv_depth = 1.0 / p_c0.norm();

    // if reprojection is too far away from target, skip this
    Eigen::Vector3d unproj_p = calib_cam.intrinsics[ref_fcid.cam_id]->unproject(ref_point).stableNormalized();
    Eigen::Vector2d reproj_p = calib_cam.intrinsics[target_fcid.cam_id]->project(T_c1_c0 * unproj_p / inv_depth);
    if(abs(reproj_p.x() - target_point.x()) > 1.0 || abs(reproj_p.y() - target_point.y()) > 1.0) {
        residual = std::numeric_limits<double>::max();
        inv_depth = -1;
        return;
    }


    residual = 0.;
    // Compute the residual with the estimated depth
    for(const auto& pattern : residual_pattern) {
        Eigen::Vector2d ref_p(ref_x + pattern.first, ref_y + pattern.second);
        Eigen::Vector3d unproj_ref_p = calib_cam.intrinsics[ref_fcid.cam_id]->unproject(ref_p).stableNormalized();
        Eigen::Vector2d target_p = calib_cam.intrinsics[target_fcid.cam_id]->project(T_c1_c0 * unproj_ref_p / inv_depth);

        if(pattern.first == 0. && pattern.second == 0.) {
            proj_x = (size_t) std::round(target_p.x());
            proj_y = (size_t) std::round(target_p.y());
        }

        if(!target_img.InBounds(target_p.x(), target_p.y(), 0.)) {
            residual = std::numeric_limits<double>::max();
            inv_depth = -1;
            return;
        }

        double ref_intensity = ref_img((size_t) std::round(ref_p.x()), (size_t) std::round(ref_p.y()));
        double target_intensity = target_img((size_t) std::round(target_p.x()), (size_t) std::round(target_p.y()));
        residual += abs((target_intensity - target_ab[1]) - exp(target_ab[0] - ref_ab[0]) * (ref_intensity - ref_ab[1]));
    }
    residual /= residual_pattern.size();
}
*/

void compute_residual(const size_t& ref_x,
                      const size_t& ref_y,
                      const double& depth,
                      const Calibration& calib_cam,
                      const FrameCamId& ref_fcid,
                      const FrameCamId& target_fcid,
                      const Sophus::SE3d& T_c1_c0,
                      const double* ref_ab,
                      const double* target_ab,
                      const pangolin::ManagedImage<uint8_t>& ref_img,
                      const pangolin::ManagedImage<uint8_t>& target_img,
                      double& residual,
                      size_t& proj_x,
                      size_t& proj_y) {

    residual = 0.;
    // Compute the residual with the estimated depth
    for(const auto& pattern : residual_pattern) {
        Eigen::Vector2d ref_p(ref_x + pattern.first, ref_y + pattern.second);

        if(!ref_img.InBounds(ref_p.x(), ref_p.y(), 0.)) {
            residual = std::numeric_limits<double>::max();
            return;
        }

        Eigen::Vector3d unproj_ref_p = calib_cam.intrinsics[ref_fcid.cam_id]->unproject(ref_p);
        Eigen::Vector2d target_p = calib_cam.intrinsics[target_fcid.cam_id]->project(T_c1_c0 * (depth * unproj_ref_p));

        if(pattern.first == 0. && pattern.second == 0.) {
            proj_x = (size_t) std::round(target_p.x());
            proj_y = (size_t) std::round(target_p.y());
        }

        if(!target_img.InBounds(target_p.x(), target_p.y(), 0.)) {
            residual = std::numeric_limits<double>::max();
            return;
        }

        double ref_intensity = ref_img((size_t) std::round(ref_p.x()), (size_t) std::round(ref_p.y()));
        double target_intensity = target_img((size_t) std::round(target_p.x()), (size_t) std::round(target_p.y()));
        residual += abs((target_intensity - target_ab[1]) - exp(target_ab[0] - ref_ab[0]) * (ref_intensity - ref_ab[1]));
    }
    residual /= residual_pattern.size();
}

/*
void compute_best_match_along_epipolar_line(const size_t& x,
                                            const size_t& y,
                                            const Calibration& calib_cam,
                                            const FrameCamId& ref_fcid,
                                            const FrameCamId& target_fcid,                                            
                                            const double* ref_ab,
                                            const double* target_ab,
                                            const pangolin::ManagedImage<uint8_t>& ref_img,
                                            const pangolin::ManagedImage<uint8_t>& target_img,
                                            size_t& best_x,
                                            size_t& best_y,
                                            double& best_residual,
                                            double& best_inv_depth) {

    const Eigen::Vector2d p((double) x, (double) y);
    const Sophus::SE3d T_c2_c1 = calib_cam.T_i_c[target_fcid.cam_id].inverse() * calib_cam.T_i_c[ref_fcid.cam_id];

    Eigen::Vector3d unproj = calib_cam.intrinsics[ref_fcid.cam_id]->unproject(p).stableNormalized();

    Eigen::Vector3d unproj_near = T_c2_c1 * (0.1 * unproj);
    Eigen::Vector3d unproj_far = T_c2_c1 * (50. * unproj);
    Eigen::Vector2d proj_near = calib_cam.intrinsics[target_fcid.cam_id]->project(unproj_near);
    Eigen::Vector2d proj_far = calib_cam.intrinsics[target_fcid.cam_id]->project(unproj_far);

    cv::Point2d p1(proj_near.x(), proj_near.y());
    cv::Point2d p2(proj_far.x(), proj_far.y());

    //cv::Point2i p_int(p.x(), p.y());
    const cv::Mat target_img_cv(target_img.h, target_img.w, CV_8U, target_img.ptr);
    cv::LineIterator it(target_img_cv, p1, p2);
    cv::Point2i best_match;

    // init best match
    best_x = target_img.w + 1;
    best_y = target_img.h + 1;
    best_residual = std::numeric_limits<double>::max();
    best_inv_depth = -1;

    for(int i=0; i < it.count; i++, ++it) {
        const cv::Point2i target_p = it.pos();

        double residual;
        double inv_depth;
        size_t target_x;
        size_t target_y;
        compute_residual(x, y, target_p.x, target_p.y, calib_cam, ref_fcid, target_fcid, T_c2_c1,
                         ref_ab, target_ab, ref_img, target_img, residual, inv_depth, target_x, target_y);

        if(inv_depth > 0. && residual < best_residual) {
            best_residual = residual;
            best_x = target_x;
            best_y = target_y;
            best_inv_depth = inv_depth;
        }
    }
}
*/

void compute_best_match_along_epipolar_line(const size_t& x,
                                            const size_t& y,
                                            const Calibration& calib_cam,
                                            const FrameCamId& ref_fcid,
                                            const FrameCamId& target_fcid,
                                            const double* ref_ab,
                                            const double* target_ab,
                                            const pangolin::ManagedImage<uint8_t>& ref_img,
                                            const pangolin::ManagedImage<uint8_t>& target_img,
                                            size_t& best_x,
                                            size_t& best_y,
                                            double& best_residual,
                                            double& second_best_residual,
                                            double& best_inv_depth) {

    const Eigen::Vector2d p((double) x, (double) y);
    const Sophus::SE3d T_c2_c1 = calib_cam.T_i_c[target_fcid.cam_id].inverse() * calib_cam.T_i_c[ref_fcid.cam_id];

    best_residual = std::numeric_limits<double>::max();
    second_best_residual = std::numeric_limits<double>::max();
    best_inv_depth = -1;

    double min_depth = 0.1;
    double max_depth = 9.;
    size_t num = 250;

    for(size_t i=0; i<num; i++) {
        double curr_depth = min_depth + i * (max_depth - min_depth) / num;
        //Eigen::Vector3d p3d_target = T_c2_c1 * (curr_depth * unproj);
        //Eigen::Vector2d p2d_target = calib_cam.intrinsics[target_fcid.cam_id]->project(p3d_target);
        double residual;
        size_t target_x;
        size_t target_y;
        compute_residual(x, y, curr_depth, calib_cam, ref_fcid, target_fcid, T_c2_c1, ref_ab, target_ab,
                         ref_img, target_img, residual, target_x, target_y);

        if(residual < best_residual) {
            second_best_residual = best_residual;
            best_residual = residual;
            best_x = target_x;
            best_y = target_y;
            best_inv_depth = 1.0 / curr_depth;
        }
        else if(residual < second_best_residual) {
            second_best_residual = residual;
        }
    }
}


// for each sample point in each image, estimates an initial depth value and computes observations
// in other frames
void pba_landmarks_from_sample_pts(PbaLandmarks& landmarks,
                                   const std::vector<CandidatePoint>& candids,
                                   const Cameras& cameras,
                                   const Calibration& calib_cam,
                                   const tbb::concurrent_unordered_map<FrameCamId, pangolin::ManagedImage<uint8_t>>& images) {

    for(const auto& candid : candids) {
        const auto& candid_fcid = candid.fcid;
        const auto& candid_x = candid.x;
        const auto& candid_y = candid.y;
        const auto& T_w_c_candid = cameras.at(candid_fcid).T_w_c;
        const auto& ref_ab = cameras.at(candid_fcid).ab;

        // Skip this candid if the residual pattern around the point is out of bounds
        if(residual_pattern_out_of_bounds(candid_x, candid_y, images.at(candid_fcid))) {
            continue;
        }

        size_t best_x;
        size_t best_y;
        double best_residual;
        double second_best_residual;
        double best_inv_depth;

        //std::cout << "finding best match" << std::endl;
        // From stereo pair, find the best match along the epipolar line, estimate depth
        const FrameCamId target_fcid(candid_fcid.frame_id, (candid_fcid.cam_id == 0) ? 1 : 0);
        //const auto& T_w_c_target = cameras.at(target_fcid).T_w_c;
        const auto& target_ab = cameras.at(target_fcid).ab;

        compute_best_match_along_epipolar_line(candid_x, candid_y, calib_cam, candid_fcid, target_fcid,
                                               ref_ab, target_ab, images.at(candid_fcid),
                                               images.at(target_fcid), best_x, best_y, best_residual, second_best_residual,
                                               best_inv_depth);

        // Skip if best_residual is not good enough
        if(best_residual > 10.) {
            continue;
        }
        if(second_best_residual < best_residual * 1.1) {
            continue;
        }

        // Estimate depth from best match
        // TODO do we need to reestimate depth?
        /*using namespace opengv;
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
        double inv_depth = 1.0 / p_c0.norm();*/

        // Find the observations in other frames
        //TODO discard obs with high photometric error
        std::vector<FrameCamId> obs;
        Eigen::Vector2d candid_point(candid_x, candid_y);
        Eigen::Vector3d p_c0 = calib_cam.intrinsics[candid_fcid.cam_id]->unproject(candid_point) / best_inv_depth;
        for(const auto& cam : cameras) {
            const auto& fcid = cam.first;
            if(fcid == candid_fcid) {
                continue;
            }
            const auto& target_img = images.at(fcid);
            const Sophus::SE3d& T_w_c = cam.second.T_w_c;
            const Eigen::Vector2d proj = calib_cam.intrinsics[fcid.cam_id]->project(T_w_c.inverse() * T_w_c_candid * p_c0);
            if(target_img.InBounds(proj.x(), proj.y(), 0.)) {
                obs.push_back(fcid);
            }
        }

        if(obs.size() == 0) {
            continue;
        }

        // Create pba landmark
        TrackId tid = get_next_track_id(landmarks);
        //std::cout << "next track id: " << tid << std::endl;
        double intensity = images.at(candid_fcid)(candid_x, candid_y);
        //std::cout << "intensity: " << intensity << std::endl;
        PbaLandmark lm(candid_fcid, candid_point, best_inv_depth, intensity, obs);
        landmarks[tid] = lm;
    }
}


// Sample points similar to DSO (outliers need to be removed after calling this function)
void sample_points(PbaLandmarks& landmarks,
                   const Calibration& calib_cam,
                   const Cameras& cameras,
                   const tbb::concurrent_unordered_map<FrameCamId, pangolin::ManagedImage<uint8_t>>& images
                   ) {

    std::random_device rnd;
    std::mt19937 rnd_generator(rnd());

    std::vector<CandidatePoint> candids;
    for(const auto& img_kv : images) {
        //std::cout << "Computing candids for image " << img_kv.first << std::endl;
        const auto& fcid = img_kv.first;
        const auto& img_raw = img_kv.second;
        const cv::Mat img(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);
        cv::Mat edges;
        cv::Canny(img, edges, 50, 150);

        // Gather edge points
        std::vector<std::pair<size_t, size_t>> edge_pts;
        for(size_t i=0; i<img_raw.w; i++) {
            for(size_t j=0; j<img_raw.h; j++) {
                if(edges.at<uint8_t>(j,i) == 255) {
                    edge_pts.push_back({i, j});
                }
            }
        }

        // Sample randomly from edge points
        std::shuffle(edge_pts.begin(), edge_pts.end(), rnd_generator);
        size_t num_samples = (size_t) 3000;
        for(size_t i=0; i<num_samples && i<edge_pts.size(); i++) {
            const size_t& candid_x = edge_pts[i].first;
            const size_t& candid_y = edge_pts[i].second;

            if(!residual_pattern_out_of_bounds(candid_x, candid_y, img_raw)) {
                candids.push_back({candid_x, candid_y, fcid});
            }
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

    //TODO debug, remove
    int count = 0;
    for(const auto& lm : landmarks) {
        if(lm.second.inv_depth == 9.0) {
            count++;
        }
    }
    std::cout << "inv depth debug count: " << count << std::endl;
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

        // skip this landmark if inverse depth is an outlier
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


void save_pba_map_file(const std::string& map_path,
                       const PbaLandmarks& landmarks,
                       const Cameras& cameras) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

void load_pba_map_file(const std::string& map_path,
                       Cameras& cameras,
                       PbaLandmarks& landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}



}


