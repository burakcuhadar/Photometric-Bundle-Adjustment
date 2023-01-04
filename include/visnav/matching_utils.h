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

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  const Eigen::Vector3d t = t_0_1.normalized();
  Eigen::Matrix3d t_0_1_skew;
  t_0_1_skew << 0, -t(2), t(1), t(2), 0, -t(0), -t(1), t(0), 0;

  E = t_0_1_skew * R_0_1;
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    Eigen::Vector3d x_L = cam1->unproject(p0_2d);
    Eigen::Vector3d x_R = cam2->unproject(p1_2d);
    if (std::abs(x_L.transpose() * E * x_R) <= epipolar_error_threshold) {
      md.inliers.emplace_back(md.matches[j].first, md.matches[j].second);
    }
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();
  md.T_i_j = Sophus::SE3d();

  // Run RANSAC with using opengv's CentralRelativePose and store
  // the final inlier indices in md.inliers and the final relative pose in
  // md.T_i_j (normalize translation). If the number of inliers is smaller than
  // ransac_min_inliers, leave md.inliers empty. Note that if the initial RANSAC
  // was successful, you should do non-linear refinement of the model parameters
  // using all inliers, and then re-estimate the inlier set with the refined
  // model parameters.

  using namespace opengv;
  // Create bearing vectors
  bearingVectors_t bearingVectors1;
  bearingVectors_t bearingVectors2;
  for (const auto& match : md.matches) {
    Eigen::Vector3d v1 = cam1->unproject(kd1.corners[match.first]);
    v1.normalize();
    bearingVectors1.push_back(v1);
    Eigen::Vector3d v2 = cam2->unproject(kd2.corners[match.second]);
    v2.normalize();
    bearingVectors2.push_back(v2);
  }

  // Setup central relative pose problem with RANSAC
  relative_pose::CentralRelativeAdapter adapter(bearingVectors1,
                                                bearingVectors2);
  sac::Ransac<sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;
  std::shared_ptr<sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new sac_problems::relative_pose::CentralRelativePoseSacProblem(
              adapter, sac_problems::relative_pose::
                           CentralRelativePoseSacProblem::NISTER));
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  // ransac.max_iterations_ = maxIterations;

  bool success = ransac.computeModel();

  // Store the inliers and relative pose
  transformation_t best_transformation = ransac.model_coefficients_;
  Eigen::Vector3d best_translation = best_transformation.block(0, 3, 3, 1);
  best_translation.normalize();
  md.T_i_j =
      Sophus::SE3d(best_transformation.block(0, 0, 3, 3), best_translation);
  if ((int)ransac.inliers_.size() > ransac_min_inliers) {
    for (size_t i = 0; i < ransac.inliers_.size(); i++) {
      md.inliers.push_back(md.matches[ransac.inliers_[i]]);
    }
  }

  // If RANSAC is succesful, refine relative pose and inliers
  if (success) {
    bearingVectors_t refine_bearingVectors1;
    bearingVectors_t refine_bearingVectors2;
    for (const auto& idx : ransac.inliers_) {
      refine_bearingVectors1.push_back(bearingVectors1[idx]);
      refine_bearingVectors2.push_back(bearingVectors2[idx]);
    }
    relative_pose::CentralRelativeAdapter refine_adapter(
        refine_bearingVectors1, refine_bearingVectors2);
    refine_adapter.sett12(best_transformation.block(0, 3, 3, 1));
    refine_adapter.setR12(best_transformation.block(0, 0, 3, 3));
    transformation_t refined_transformation =
        relative_pose::optimize_nonlinear(refine_adapter);

    // Update the set of inliners using the refined relative pose
    std::shared_ptr<sac_problems::relative_pose::CentralRelativePoseSacProblem>
        refine_relposeproblem_ptr(
            new sac_problems::relative_pose::CentralRelativePoseSacProblem(
                refine_adapter, sac_problems::relative_pose::
                                    CentralRelativePoseSacProblem::NISTER));
    std::vector<int> refined_inliers;
    refine_relposeproblem_ptr->selectWithinDistance(
        refined_transformation, ransac_thresh, refined_inliers);

    // Store the refined inliers and relative pose
    Eigen::Vector3d refined_translation =
        refined_transformation.block(0, 3, 3, 1);
    refined_translation.normalize();
    md.T_i_j = Sophus::SE3d(refined_transformation.block(0, 0, 3, 3),
                            refined_translation);
    md.inliers.clear();
    if ((int)refined_inliers.size() > ransac_min_inliers) {
      for (const auto& idx : refined_inliers) {
        md.inliers.push_back(md.matches[ransac.inliers_[idx]]);
      }
    }
  }
}
}  // namespace visnav
