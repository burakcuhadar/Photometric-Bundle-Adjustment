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

#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

template <class T>
class AbstractCamera;

struct ReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                          const Eigen::Vector3d& p_3d,
                          const std::string& cam_model)
      : p_2d(p_2d), p_3d(p_3d), cam_model(cam_model) {}

  template <class T>
  bool operator()(T const* const sT_w_i, T const* const sT_i_c,
                  T const* const sIntr, T* sResiduals) const {
    Eigen::Map<Sophus::SE3<T> const> const T_w_i(sT_w_i);
    Eigen::Map<Sophus::SE3<T> const> const T_i_c(sT_i_c);

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);

    residuals = p_2d - cam->project(T_i_c.inverse() * T_w_i.inverse() * p_3d);

    return true;
  }

  Eigen::Vector2d p_2d;
  Eigen::Vector3d p_3d;
  std::string cam_model;
};

struct BundleAdjustmentReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                                          const Eigen::Vector2d& p_2d_ref,
                                          double* ref_intrinsics,
                                          const std::string& cam_model)
      : p_2d(p_2d), p_2d_ref(p_2d_ref), ref_intrinsics(ref_intrinsics), cam_model(cam_model) {}

  template <class T>
  bool operator()(T const* const sT_w_c1, T const* const sT_w_c2,
                  T const* const inv_depth, T const* const sIntr_c2,
                  T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const T_w_c1(sT_w_c1);
    Eigen::Map<Sophus::SE3<T> const> const T_w_c2(sT_w_c2);
    //Eigen::Map<Eigen::Matrix<T, 3, 1> const> const p_3d_w(sp_3d_w);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    //const std::shared_ptr<AbstractCamera<T>> cam1 =
    //    AbstractCamera<T>::from_data(cam_model, sIntr_c1);
    T ref_intrinsics_[8];
    for(int i=0; i<8; i++) {
        ref_intrinsics_[i] = T(ref_intrinsics[i]);
    }
    const std::shared_ptr<AbstractCamera<T>> cam1 =
        AbstractCamera<T>::from_data(cam_model, ref_intrinsics_);
    const std::shared_ptr<AbstractCamera<T>> cam2 =
        AbstractCamera<T>::from_data(cam_model, sIntr_c2);


    // Compute reprojection error
    //residuals = p_2d - cam->project(T_w_c.inverse() * p_3d_w);
    const Eigen::Matrix<T, 2, 1> p_2d_ref_ = p_2d_ref.cast<T>();
    Eigen::Matrix<T, 3, 1> unproj_p_2d_ref = cam1->unproject(p_2d_ref_);
    unproj_p_2d_ref.normalize();
    residuals = p_2d - cam2->project(T_w_c2.inverse() * T_w_c1 * (unproj_p_2d_ref / inv_depth[0]));


    return true;
  }

  Eigen::Vector2d p_2d;
  Eigen::Vector2d p_2d_ref;
  double* ref_intrinsics;
  std::string cam_model;
};

}  // namespace visnav
