#pragma once

#include <ceres/ceres.h>
#include <visnav/common_types.h>
#include <visnav/pba_types.h>
#include <visnav/photometric.h>
#include <thread>

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

      problem.AddParameterBlock(&cam.second.a, 1);
      problem.AddParameterBlock(&cam.second.b, 1);
      //TODO required?
      problem.SetParameterLowerBound(&cam.second.a, 0, -5.0);
      problem.SetParameterUpperBound(&cam.second.a, 0, 5.0);

      if (!options.optimize_btf_params) {
          problem.SetParameterBlockConstant(&cam.second.a);
          problem.SetParameterBlockConstant(&cam.second.b);
      }

      if(*fixed_cameras.begin() == cam.first) {
          std::cout << "fixing btf parameters of " << cam.first << std::endl;
          problem.SetParameterBlockConstant(&cam.second.a);
          problem.SetParameterBlockConstant(&cam.second.b);
      }
    }

    //TODO required?
    for(auto& landmark_kv : landmarks) {
        problem.AddParameterBlock(&landmark_kv.second.inv_depth, 1);
        problem.SetParameterLowerBound(&landmark_kv.second.inv_depth, 0, 1.0/20.0);
        problem.SetParameterUpperBound(&landmark_kv.second.inv_depth, 0, 1e8);
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
        const auto& ref_img = images.at(ref_fcid);

        // compute intensities of the pixels in the reference frame
        // also check whether all points of the pattern in the image, skip this landmark if not
        bool in_image = true;
        //ceres::Grid2D<uint8_t,1> ref_grid(ref_img.begin(), 0, h, 0, w);
        const auto ref_grid = *(grids.at(ref_fcid));
        ceres::BiCubicInterpolator<ceres::Grid2D<double,1>> ref_interp(ref_grid);
        std::vector<double> intensities;
        std::vector<Eigen::Vector2d> points;
        std::vector<double> grad_norms;
        for(const auto& pattern : residual_pattern) {
            Eigen::Vector2d p(lm.p_2d(0) + pattern.first, lm.p_2d(1) + pattern.second);
            points.push_back(p);
            if(p(0) < 0 || p(0) > w || p(1) < 0 || p(1) > h) {
                in_image = false;
                break;
            }
            //TODO check also whether in target image?
            double intensity;
            double dIdx;
            double dIdy;
            ref_interp.Evaluate(p(1), p(0), &intensity, &dIdy, &dIdx);
            //intensities.push_back(ref_img(size_t(p(0)), size_t(p(1)))); //TODO this or with interp?
            intensities.push_back(intensity);
            grad_norms.push_back(dIdx*dIdx + dIdy*dIdy);
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
            //ceres::Grid2D<uint8_t,1> grid(target_img.begin(), 0, target_img.h, 0, target_img.w);
            const auto grid = *(grids.at(target_fcid));
            ceres::BiCubicInterpolator<ceres::Grid2D<double,1>> interp(grid);


            for(size_t i=0; i<pattern_size; i++) {
                PhotometricCostFunctor* cost_functor = new PhotometricCostFunctor(
                            points[i],
                            intensities[i],
                            ref_intrinsics->name(),
                            interp,
                            h,
                            w);

                ceres::HuberLoss* huber = new ceres::HuberLoss(options.huber_parameter);
                ceres::ScaledLoss* scaled_huber = new ceres::ScaledLoss(huber, 1./(1.+grad_norms[i]), ceres::DO_NOT_TAKE_OWNERSHIP);


                if(target_fcid.cam_id == ref_fcid.cam_id) {
                    ceres::CostFunction* cost_func = new ceres::AutoDiffCostFunction<PhotometricCostFunctor,
                            1, Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 8, 1, 1, 1, 1, 1>(cost_functor);


                    problem.AddResidualBlock(
                                cost_func,
                                scaled_huber,
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
                            1, Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 8, 8, 1, 1, 1, 1, 1>(cost_functor);

                    problem.AddResidualBlock(
                                cost_func,
                                scaled_huber,
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


