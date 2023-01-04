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

#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // Project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.
  for (const auto& landmark_kv : landmarks) {
    const auto& track_id = landmark_kv.first;
    const auto& landmark = landmark_kv.second;

    // Landmark position in camera frame
    const Eigen::Vector3d landmark_p_c = current_pose.inverse() * landmark.p;
    // Check whether landmark is behind camera
    if (landmark_p_c(2) < cam_z_threshold) {
      continue;
    }

    const Eigen::Vector2d proj = cam->project(landmark_p_c);
    // Check whether projection is outside of the image
    if (proj(0) < 0 || proj(1) < 0 || proj(0) > cam->width() ||
        proj(1) > cam->height()) {
      continue;
    }

    projected_points.push_back(proj);
    projected_track_ids.push_back(track_id);
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_threshold,
    const double feature_match_dist_2_best, LandmarkMatchData& md) {
  md.matches.clear();

  // Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_threshold and feature_match_dist_2_best
  // should be used to filter outliers the same way as in exercise 3. You should
  // fill md.matches with <featureId,trackId> pairs for the successful matches
  // that pass all tests.
  for (size_t feature_id = 0; feature_id < kdl.corners.size(); feature_id++) {
    const auto& kp_desc = kdl.corner_descriptors[feature_id];

    int smallest_dist = 257;
    int second_smallest_dist = 257;
    TrackId best_match = -1;

    // search for matches inside a circle with radius match_max_dist_2d around
    // the point location
    for (size_t proj_idx = 0; proj_idx < projected_points.size(); proj_idx++) {
      const auto dist_2d =
          (kdl.corners[feature_id] - projected_points[proj_idx]).norm();

      int best_dist = 257;

      if (dist_2d <= match_max_dist_2d) {
        for (const auto& feature_track_kv :
             landmarks.at(projected_track_ids[proj_idx]).obs) {
          const auto& frame_cam_id = feature_track_kv.first;
          const auto& obs_feature_id = feature_track_kv.second;
          const auto& obs_desc = feature_corners.at(frame_cam_id)
                                     .corner_descriptors[obs_feature_id];
          int desc_dist = (kp_desc ^ obs_desc).count();
          if (desc_dist < best_dist) {
            best_dist = desc_dist;
          }
        }
      }

      if (best_dist < smallest_dist) {
        second_smallest_dist = smallest_dist;
        smallest_dist = best_dist;
        best_match = projected_track_ids[proj_idx];
      } else if (best_dist < second_smallest_dist) {
        second_smallest_dist = best_dist;
      }
    }

    if (smallest_dist >= feature_match_threshold) {
      continue;
    }
    if (second_smallest_dist < smallest_dist * feature_match_dist_2_best) {
      continue;
    }
    if (best_match != -1) {
      md.matches.push_back(std::make_pair(feature_id, best_match));
    }
  }
}

void localize_camera(const Sophus::SE3d& current_pose,
                     const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     LandmarkMatchData& md) {
  md.inliers.clear();

  // default to previous pose if not enough inliers
  md.T_w_c = current_pose;

  if (md.matches.size() < 4) {
    return;
  }

  // Find the pose (md.T_w_c) and the inliers (md.inliers) using
  // the landmark to keypoints matches and PnP. This should be similar to the
  // localize_camera in exercise 4 but in this exercise we don't explicitly have
  // tracks.
  using namespace opengv;

  bearingVectors_t bearingVectors;
  points_t points;

  for (const auto& match : md.matches) {
    const auto& feature_id = match.first;
    Eigen::Vector3d v = cam->unproject(kdl.corners[feature_id]);
    v.normalize();
    bearingVectors.push_back(v);
    points.push_back(landmarks.at(match.second).p);
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
  md.T_w_c = Sophus::SE3d(nonlinear_transformation.block(0, 0, 3, 3),
                          nonlinear_transformation.block(0, 3, 3, 1));

  for (const auto& idx : ransac.inliers_) {
    md.inliers.push_back(md.matches[idx]);
  }
}

void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, const MatchData& md_stereo,
                       const LandmarkMatchData& md, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains feature to landmark
  // matches for the left camera (camera 0). For all inlier feature to landmark
  // matches add the observations to the existing landmarks. If the left
  // camera's feature appears also in md_stereo.inliers, then add both
  // observations. For all inlier stereo observations that were not added to the
  // existing landmarks, triangulate and add new landmarks. Here
  // next_landmark_id is a running index of the landmarks, so after adding a new
  // landmark you should always increase next_landmark_id by 1.

  std::set<FeatureId> added;
  // For all inlier landmark matches add observations for the left camera to the
  // existing landmarks
  for (const auto& match : md.inliers) {
    const auto& feature_id = match.first;
    const auto& track_id = match.second;

    landmarks.at(track_id).obs[fcidl] = feature_id;
    added.insert(feature_id);
    // If the left camera's feature appears also in md_stereo.inliers, then add
    // both observations.
    for (const auto& stereo_match : md_stereo.inliers) {
      if (stereo_match.first == feature_id) {
        landmarks.at(track_id).obs[fcidr] = stereo_match.second;
      }
    }
  }

  // For all inlier stereo observations that were not added to the existing
  // landmarks, triangulate and add new landmarks.
  using namespace opengv;
  bearingVectors_t l_bearingVectors;
  bearingVectors_t r_bearingVectors;
  std::vector<FeatureId> l_feature_ids;
  std::vector<FeatureId> r_feature_ids;
  const auto& cam_l = calib_cam.intrinsics[fcidl.cam_id];
  const auto& cam_r = calib_cam.intrinsics[fcidr.cam_id];

  for (const auto& stereo_match : md_stereo.inliers) {
    const auto& left_fid = stereo_match.first;
    const auto& right_fid = stereo_match.second;
    if (added.count(left_fid) == 0) {
      Eigen::Vector3d bearing_l = cam_l->unproject(kdl.corners[left_fid]);
      bearing_l.normalize();
      Eigen::Vector3d bearing_r = cam_r->unproject(kdr.corners[right_fid]);
      bearing_r.normalize();
      l_bearingVectors.push_back(bearing_l);
      r_bearingVectors.push_back(bearing_r);
      l_feature_ids.push_back(left_fid);
      r_feature_ids.push_back(right_fid);
    }
  }

  relative_pose::CentralRelativeAdapter adapter(l_bearingVectors,
                                                r_bearingVectors, t_0_1, R_0_1);

  // Triangulate
  for (size_t i = 0; i < l_bearingVectors.size(); i++) {
    Eigen::Vector3d p = md.T_w_c * triangulation::triangulate(adapter, i);
    FeatureTrack obs;
    FeatureTrack outlier_obs;
    obs[fcidl] = l_feature_ids[i];
    obs[fcidr] = r_feature_ids[i];
    landmarks[next_landmark_id++] = {p, obs, outlier_obs};
  }
}

void remove_old_keyframes(const FrameCamId fcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(fcidl.frame_id);

  // Remove old cameras and observations if the number of keyframe
  // pairs (left and right image is a pair) is larger than max_num_kfs. The ids
  // of all the keyframes that are currently in the optimization should be
  // stored in kf_frames. Removed keyframes should be removed from cameras and
  // landmarks with no left observations should be moved to old_landmarks.
  auto kf_frames_itr = kf_frames.cbegin();
  while (kf_frames_itr != kf_frames.cend() &&
         (int)kf_frames.size() > max_num_kfs) {
    FrameId removed_frameid = *kf_frames_itr;
    // kf_frames.erase(removed_frameid);
    kf_frames_itr = kf_frames.erase(kf_frames_itr);
    const auto left_cam = FrameCamId(removed_frameid, 0);
    const auto right_cam = FrameCamId(removed_frameid, 1);
    cameras.erase(left_cam);
    cameras.erase(right_cam);

    for (auto& lm_kv : landmarks) {
      lm_kv.second.obs.erase(left_cam);
      lm_kv.second.obs.erase(right_cam);
    }

    for (auto it = landmarks.cbegin(); it != landmarks.cend();) {
      const auto& obs = it->second.obs;
      const auto track_id = it->first;
      const auto lm = it->second;

      if (obs.size() == 0) {
        it = landmarks.erase(it);
        old_landmarks[track_id] = lm;
      } else {
        it++;
      }
    }
  }
}
}  // namespace visnav
