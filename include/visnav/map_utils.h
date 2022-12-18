/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <fstream>
#include <thread>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

#include <visnav/reprojection.h>
#include <visnav/local_parameterization_se3.hpp>

#include <visnav/tracks.h>

namespace visnav {

// save map with all features and matches
void save_map_file(const std::string& map_path, const Corners& feature_corners,
                   const Matches& feature_matches,
                   const FeatureTracks& feature_tracks,
                   const FeatureTracks& outlier_tracks, const Cameras& cameras,
                   const Landmarks& landmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
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

// load map with all features and matches
void load_map_file(const std::string& map_path, Corners& feature_corners,
                   Matches& feature_matches, FeatureTracks& feature_tracks,
                   FeatureTracks& outlier_tracks, Cameras& cameras,
                   Landmarks& landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
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

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const FrameCamId& fcid0,
                                   const FrameCamId& fcid1,
                                   const Calibration& calib_cam,
                                   const Corners& feature_corners,
                                   const FeatureTracks& feature_tracks,
                                   const Cameras& cameras,
                                   Landmarks& landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<FrameCamId> fcids = {fcid0, fcid1};
  if (!GetTracksInImages(fcids, feature_tracks, shared_track_ids)) {
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  std::vector<TrackId> new_track_ids;

  // Triangulate all new features and add to the map

  using namespace opengv;

  // Create bearing vectors for the features
  bearingVectors_t bearingVectors0;
  bearingVectors_t bearingVectors1;
  for (const auto& track_id : shared_track_ids) {
    if (landmarks.count(track_id) == 0) {
      const FeatureId& feature_id0 = feature_tracks.at(track_id).at(fcid0);
      const FeatureId& feature_id1 = feature_tracks.at(track_id).at(fcid1);
      const Eigen::Vector2d& feature0 =
          feature_corners.at(fcid0).corners[feature_id0];
      const Eigen::Vector2d& feature1 =
          feature_corners.at(fcid1).corners[feature_id1];
      const auto& cam0 = calib_cam.intrinsics[fcid0.cam_id];
      const auto& cam1 = calib_cam.intrinsics[fcid1.cam_id];
      Eigen::Vector3d v0 = cam0->unproject(feature0);
      v0.normalize();
      Eigen::Vector3d v1 = cam1->unproject(feature1);
      v1.normalize();
      bearingVectors0.push_back(v0);
      bearingVectors1.push_back(v1);
      new_track_ids.push_back(track_id);
    }
  }

  const Sophus::SE3d& T_w_c0 = cameras.at(fcid0).T_w_c;
  const Sophus::SE3d& T_w_c1 = cameras.at(fcid1).T_w_c;
  const Sophus::SE3d T_c0_c1 = T_w_c0.inverse() * T_w_c1;
  const Eigen::Matrix3d R01 = T_c0_c1.rotationMatrix();
  const Eigen::Vector3d t01 = T_c0_c1.translation();
  relative_pose::CentralRelativeAdapter adapter(bearingVectors0,
                                                bearingVectors1, t01, R01);

  // Triangulate
  for (size_t i = 0; i < bearingVectors0.size(); i++) {
    Eigen::Vector3d p = triangulation::triangulate(adapter, i);

    // Fill obs and outlier_obs
    const FeatureTrack& track = feature_tracks.at(new_track_ids[i]);
    FeatureTrack obs;
    FeatureTrack outlier_obs;
    for (const auto& track_kv : track) {
      if (cameras.count(track_kv.first) > 0) {
        obs[track_kv.first] = track_kv.second;
      }
    }

    double inv_depth = 1.0 / p.norm(); //TODO check correctness?
    landmarks[new_track_ids[i]] = {inv_depth, obs, outlier_obs};
  }

  return new_track_ids.size();
}

// Initialize the scene from a stereo pair, using the known transformation from
// camera calibration. This adds the inital two cameras and triangulates shared
// landmarks.
// Note: in principle we could also initialize a map from another images pair
// using the transformation from the pairwise matching with the 5-point
// algorithm. However, using a stereo pair has the advantage that the map is
// initialized with metric scale.
bool initialize_scene_from_stereo_pair(const FrameCamId& fcid0,
                                       const FrameCamId& fcid1,
                                       const Calibration& calib_cam,
                                       const Corners& feature_corners,
                                       const FeatureTracks& feature_tracks,
                                       Cameras& cameras, Landmarks& landmarks) {
  // check that the two image ids refer to a stereo pair
  if (!(fcid0.frame_id == fcid1.frame_id && fcid0.cam_id != fcid1.cam_id)) {
    std::cerr << "Images " << fcid0 << " and " << fcid1
              << " don't form a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  // Initialize scene (add initial cameras and landmarks)
  Camera left_cam{Sophus::SE3d()};
  cameras[fcid0] = left_cam;
  Camera right_cam{calib_cam.T_i_c[fcid1.cam_id]};
  cameras[fcid1] = right_cam;

  add_new_landmarks_between_cams(fcid0, fcid1, calib_cam, feature_corners,
                                 feature_tracks, cameras, landmarks);

  return true;
}

// Localize a new camera in the map given a set of observed landmarks. We use
// pnp and ransac to localize the camera in the presence of outlier tracks.
// After finding an inlier set with pnp, we do non-linear refinement using all
// inliers and also update the set of inliers using the refined pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv documentation on
// how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html#sec_threshold
void localize_camera(
    const FrameCamId& fcid, const std::vector<TrackId>& shared_track_ids,
    const Calibration& calib_cam, const Corners& feature_corners, const Cameras& cameras,
    const FeatureTracks& feature_tracks, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d& T_w_c, std::vector<TrackId>& inlier_track_ids) {
  inlier_track_ids.clear();

  // Localize a new image in a given map
  using namespace opengv;

  bearingVectors_t bearingVectors;
  const auto& cam = calib_cam.intrinsics[fcid.cam_id];
  for (const auto& id : shared_track_ids) {
    const Eigen::Vector2d& corner =
        feature_corners.at(fcid).corners[feature_tracks.at(id).at(fcid)];
    Eigen::Vector3d v = cam->unproject(corner);
    v.normalize();
    bearingVectors.push_back(v);
  }

  opengv::points_t points;
  for (const auto& id : shared_track_ids) {
    points.push_back(landmarks.at(id).get_p(cameras, calib_cam, feature_corners));
  }

  absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);
  sac::Ransac<sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
  std::shared_ptr<sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter,
              sac_problems::absolute_pose::AbsolutePoseSacProblem::EPNP));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ =
      1.0 - cos(atan(reprojection_error_pnp_inlier_threshold_pixel / 500.0));
  ransac.computeModel();

  transformation_t ransac_transformation = ransac.model_coefficients_;

  // Non-linear refinement
  bearingVectors_t inlier_bearingVectors;
  points_t inlier_points;
  for (const auto& idx : ransac.inliers_) {
    inlier_bearingVectors.push_back(bearingVectors[idx]);
    inlier_points.push_back(points[idx]);
  }

  absolute_pose::CentralAbsoluteAdapter refine_adapter(inlier_bearingVectors,
                                                       inlier_points);
  refine_adapter.sett(ransac_transformation.block(0, 3, 3, 1));
  refine_adapter.setR(ransac_transformation.block(0, 0, 3, 3));
  transformation_t nonlinear_transformation =
      absolute_pose::optimize_nonlinear(refine_adapter);

  T_w_c = Sophus::SE3d(nonlinear_transformation.block(0, 0, 3, 3),
                       nonlinear_transformation.block(0, 3, 3, 1));
  for (const auto& idx : ransac.inliers_) {
    inlier_track_ids.push_back(shared_track_ids[idx]);
  }
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void bundle_adjustment(const Corners& feature_corners,
                       const BundleAdjustmentOptions& options,
                       const std::set<FrameCamId>& fixed_cameras,
                       Calibration& calib_cam, Cameras& cameras,
                       Landmarks& landmarks) {
  ceres::Problem problem;

  // Setup optimization problem
  for (auto& cam : cameras) {
    problem.AddParameterBlock(cam.second.T_w_c.data(),
                              Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);
    if (fixed_cameras.count(cam.first) > 0) {
      problem.SetParameterBlockConstant(cam.second.T_w_c.data());
    }
  }

  //TODO optimization of intrinsics doesn't work correctly, ref frame intrinsics are kept fixed
  if (!options.optimize_intrinsics) {
    for (const auto& intrinsics : calib_cam.intrinsics) {
      problem.AddParameterBlock(intrinsics->data(), 8);
      problem.SetParameterBlockConstant(intrinsics->data());
    }
  }

  for (auto& landmark_kv : landmarks) {
    // const auto& track_id = landmark_kv.first;
    auto& landmark = landmark_kv.second;
    //std::cout << "landmark inv depth before: " << landmark.inv_depth << std::endl; //TODO remove
    const auto& ref_frame_cam_id = landmark.obs.begin()->first;
    const auto& ref_feature_id = landmark.obs.begin()->second;
    // start iterating from the second observation, the first frame is used as anchor
    //for (auto& feature_track_kv : landmark.obs) {
    for(auto it = std::next(landmark.obs.begin()); it != landmark.obs.end(); ++it) {
      const auto& frame_cam_id = it->first;
      const auto& feature_id = it->second;

      BundleAdjustmentReprojectionCostFunctor* cost_functor =
          new BundleAdjustmentReprojectionCostFunctor(
              feature_corners.at(frame_cam_id).corners[feature_id],
              feature_corners.at(ref_frame_cam_id).corners[ref_feature_id],
              calib_cam.intrinsics[ref_frame_cam_id.cam_id]->data(),
              calib_cam.intrinsics[ref_frame_cam_id.cam_id]->name());
      ceres::CostFunction* cost_func = new ceres::AutoDiffCostFunction<
          BundleAdjustmentReprojectionCostFunctor, 2,
          Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 1, 8>(cost_functor);
      problem.AddResidualBlock(
          cost_func,
          options.use_huber ? new ceres::HuberLoss(options.huber_parameter)
                            : nullptr,
          cameras.at(ref_frame_cam_id).T_w_c.data(), cameras.at(frame_cam_id).T_w_c.data(), &landmark.inv_depth,
          calib_cam.intrinsics[frame_cam_id.cam_id]->data());
    }
  }

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
  //TODO remove below
  /*for (auto& landmark_kv : landmarks) {
    // const auto& track_id = landmark_kv.first;
    auto& landmark = landmark_kv.second;
    std::cout << "landmark inv depth after: " << landmark.inv_depth << std::endl;
  }*/
}

}  // namespace visnav
