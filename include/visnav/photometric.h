#pragma once

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <memory>
#include <ceres/cubic_interpolation.h>
#include <ceres/ceres.h>
#include <pangolin/image/managed_image.h>
#include <visnav/common_types.h>

namespace visnav {

template <class T>
class AbstractCamera;

const size_t pattern_size = 8;
const std::vector<std::pair<double, double>> residual_pattern = {{0.,0.},
                                                                 {0.,2.}, {-1.,1.}, {-2.,0.},
                                                                 {-1.,-1.}, {0.,-2.}, {1.,-1.},
                                                                 {2.,0.}};

struct PhotometricCostFunctor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PhotometricCostFunctor(const Eigen::Vector2d p_2d,
                           const double& intensity_ref,
                           const std::string cam_model,
                           const ceres::BiCubicInterpolator<ceres::Grid2D<double,1>>& interp_target,
                           const size_t h, const size_t w)
        : p_2d(p_2d), intensity_ref(intensity_ref), cam_model(cam_model), interp_target(interp_target),
          h(h), w(w) {}

    template <class T>
    Eigen::Matrix<T, 2, 1> project_to_target(const Eigen::Matrix<T, 2, 1>& p,
                                             const std::shared_ptr<AbstractCamera<T>>& cam1,
                                             const std::shared_ptr<AbstractCamera<T>>& cam2,
                                             const Eigen::Map<Sophus::SE3<T> const>& T_w_c1,
                                             const Eigen::Map<Sophus::SE3<T> const>& T_w_c2,
                                             const T* const inv_depth) const {

        // project p into the target frame(cam2)
        Eigen::Matrix<T, 3, 1> p_unproj = cam1->unproject(p).stableNormalized();
        Eigen::Matrix<T, 2, 1> p_target = cam2->project(T_w_c2.inverse() * T_w_c1 * (p_unproj / inv_depth[0]));

        return p_target;
    }


    template <class T>
    bool operator()(T const* const sT_w_c1, T const* const sT_w_c2,
                    T const* const sIntr1, T const* const sIntr2,
                    T const* const a1, T const* const b1,
                    T const* const a2, T const* const b2,
                    T const* const inv_depth,
                    T* residuals) const {

        if(inv_depth[0] < T(0.)) {
            residuals[0] = T(0.);
            return true;
        }

        // map inputs
        Eigen::Map<Sophus::SE3<T> const> const T_w_c1(sT_w_c1);
        Eigen::Map<Sophus::SE3<T> const> const T_w_c2(sT_w_c2);

        const std::shared_ptr<AbstractCamera<T>> cam1 =
            AbstractCamera<T>::from_data(cam_model, sIntr1);
        const std::shared_ptr<AbstractCamera<T>> cam2 =
            AbstractCamera<T>::from_data(cam_model, sIntr2);


        Eigen::Matrix<T, 2, 1> p = p_2d.cast<T>();
        Eigen::Matrix<T, 2, 1> p_target = project_to_target(p, cam1, cam2, T_w_c1, T_w_c2, inv_depth);
        if(p_target(0) < T(0.) || p_target(0) > T(w) || p_target(1) < T(0.) || p_target(1) > T(h)) {
            residuals[0] = T(0.);
            return true;
        }

        T intensity_target;
        interp_target.Evaluate(p_target(1), p_target(0), &intensity_target);
        residuals[0] = (intensity_target - b2[0]) - exp(a2[0] - a1[0]) * (T(intensity_ref) - b1[0]);

        return true;
    }


    template <class T>
    bool operator()(T const* const sT_w_c1, T const* const sT_w_c2,
                    T const* const sIntr,
                    T const* const a1, T const* const b1,
                    T const* const a2, T const* const b2,
                    T const* const inv_depth,
                    T* residuals) const {

        if(inv_depth[0] < T(0.)) {
            residuals[0] = T(0.);
            return true;
        }

        // map inputs
        Eigen::Map<Sophus::SE3<T> const> const T_w_c1(sT_w_c1);
        Eigen::Map<Sophus::SE3<T> const> const T_w_c2(sT_w_c2);

        const std::shared_ptr<AbstractCamera<T>> cam =
            AbstractCamera<T>::from_data(cam_model, sIntr);

        Eigen::Matrix<T, 2, 1> p = p_2d.cast<T>();
        Eigen::Matrix<T, 2, 1> p_target = project_to_target(p, cam, cam, T_w_c1, T_w_c2, inv_depth);
        if(p_target(0) < T(0.) || p_target(0) > T(w) || p_target(1) < T(0.) || p_target(1) > T(h)) {
            residuals[0] = T(0.);
            return true;
        }

        T intensity_target;
        interp_target.Evaluate(p_target(1), p_target(0), &intensity_target);
        residuals[0] = (intensity_target - b2[0]) - exp(a2[0] - a1[0]) * (T(intensity_ref) - b1[0]);

        return true;
    }

    const Eigen::Vector2d p_2d;
    const double intensity_ref;
    //const std::vector<uint8_t>& intensities;
    const std::string cam_model;
    const ceres::BiCubicInterpolator<ceres::Grid2D<double,1>>& interp_target;
    const size_t h;
    const size_t w;
};



/*
//TODO no gradient weighting!
struct PhotometricCostFunctor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PhotometricCostFunctor(const Eigen::Vector2d p_2d,
                           const std::vector<uint8_t>& intensities,
                           const std::string cam_model,
                           const ceres::BiCubicInterpolator<ceres::Grid2D<uint8_t,1>>& interp_target,
                           const size_t h, const size_t w)
        : p_2d(p_2d), intensities(intensities), cam_model(cam_model), interp_target(interp_target),
          h(h), w(w) {}

    template <class T>
    Eigen::Matrix<T, 2, 1> project_to_target(const Eigen::Matrix<T, 2, 1>& p,
                                             const std::shared_ptr<AbstractCamera<T>>& cam1,
                                             const std::shared_ptr<AbstractCamera<T>>& cam2,
                                             const Eigen::Map<Sophus::SE3<T> const>& T_w_c1,
                                             const Eigen::Map<Sophus::SE3<T> const>& T_w_c2,
                                             const T* const inv_depth) const {

        // project p into the target frame(cam2)
        Eigen::Matrix<T, 3, 1> p_unproj = cam1->unproject(p);
        p_unproj.normalize(); //TODO needed?
        Eigen::Matrix<T, 2, 1> p_target = cam2->project(T_w_c2.inverse() * T_w_c1 * (p_unproj / inv_depth[0]));

        return p_target;
    }


    template <class T>
    bool operator()(T const* const sT_w_c1, T const* const sT_w_c2,
                    T const* const sIntr1, T const* const sIntr2,
                    T const* const a1, T const* const b1,
                    T const* const a2, T const* const b2,
                    T const* const inv_depth,
                    T* sResiduals) const {


        Eigen::Map<Eigen::Matrix<T, pattern_size, 1>> residuals(sResiduals);

        if(inv_depth[0] < T(1e-5) || inv_depth[0] > T(15)) {
            residuals = Eigen::Matrix<T, pattern_size, 1>::Zero();
            return true;
        }

        // map inputs
        Eigen::Map<Sophus::SE3<T> const> const T_w_c1(sT_w_c1);
        Eigen::Map<Sophus::SE3<T> const> const T_w_c2(sT_w_c2);

        const std::shared_ptr<AbstractCamera<T>> cam1 =
            AbstractCamera<T>::from_data(cam_model, sIntr1);
        const std::shared_ptr<AbstractCamera<T>> cam2 =
            AbstractCamera<T>::from_data(cam_model, sIntr2);


        Eigen::Matrix<T, 2, 1> p_2d_ = p_2d.cast<T>();
        std::vector<Eigen::Matrix<T, 2, 1>> targets;

        for(const auto& pattern : residual_pattern) {
            Eigen::Matrix<T, 2, 1> p(p_2d_(0) + T(pattern.first), p_2d_(1) + T(pattern.second));

            Eigen::Matrix<T, 2, 1> p_target = project_to_target(p, cam1, cam2, T_w_c1, T_w_c2, inv_depth);

            if(p_target(0) < T(0.) || p_target(0) > T(w) || p_target(1) < T(0.) || p_target(1) > T(h)) {
                residuals = Eigen::Matrix<T, pattern_size, 1>::Zero();
                return true;
            }

            targets.push_back(p_target);
        }


        for(size_t i=0; i<pattern_size; i++) {

            auto& p_target = targets[i];

            T intensity_target;
            interp_target.Evaluate(p_target(1), p_target(0), &intensity_target);


            residuals[i] = (intensity_target - b2[0]) - exp(a2[0]) / exp(a1[0]) * (T(intensities[i]) - b1[0]);
        }

        return true;
    }


    template <class T>
    bool operator()(T const* const sT_w_c1, T const* const sT_w_c2,
                    T const* const sIntr,
                    T const* const a1, T const* const b1,
                    T const* const a2, T const* const b2,
                    T const* const inv_depth,
                    T* sResiduals) const {

        Eigen::Map<Eigen::Matrix<T, pattern_size, 1>> residuals(sResiduals);

        if(inv_depth[0] < T(1e-5) || inv_depth[0] > T(15)) {
            residuals = Eigen::Matrix<T, pattern_size, 1>::Zero();
            return true;
        }

        // map inputs
        Eigen::Map<Sophus::SE3<T> const> const T_w_c1(sT_w_c1);
        Eigen::Map<Sophus::SE3<T> const> const T_w_c2(sT_w_c2);

        const std::shared_ptr<AbstractCamera<T>> cam =
            AbstractCamera<T>::from_data(cam_model, sIntr);


        Eigen::Matrix<T, 2, 1> p_2d_ = p_2d.cast<T>();
        std::vector<Eigen::Matrix<T, 2, 1>> targets;

        for(const auto& pattern : residual_pattern) {
            Eigen::Matrix<T, 2, 1> p(p_2d_(0) + T(pattern.first), p_2d_(1) + T(pattern.second));

            Eigen::Matrix<T, 2, 1> p_target = project_to_target(p, cam, cam, T_w_c1, T_w_c2, inv_depth);

            if(p_target(0) < T(0.) || p_target(0) > T(w) || p_target(1) < T(0.) || p_target(1) > T(h)) {
                residuals = Eigen::Matrix<T, pattern_size, 1>::Zero();
                return true;
            }

            targets.push_back(p_target);
        }


        for(size_t i=0; i<pattern_size; i++) {

            auto& p_target = targets[i];

            T intensity_target;
            interp_target.Evaluate(p_target(1), p_target(0), &intensity_target);

            residuals[i] = (intensity_target - b2[0]) - exp(a2[0]) / exp(a1[0]) * (T(intensities[i]) - b1[0]);
        }


        return true;
    }

    const Eigen::Vector2d p_2d;
    //const double intensity_ref;
    const std::vector<uint8_t>& intensities;
    const std::string cam_model;
    const ceres::BiCubicInterpolator<ceres::Grid2D<uint8_t,1>>& interp_target;
    const size_t h;
    const size_t w;
};
*/

}



/*
struct BundleAdjustmentReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                                          const std::string& cam_model)
      : p_2d(p_2d), cam_model(cam_model) {}

  template <class T>
  bool operator()(T const* const sT_w_c, T const* const sp_3d_w,
                  T const* const sIntr, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const T_w_c(sT_w_c);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const p_3d_w(sp_3d_w);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);

    // Compute reprojection error
    residuals = p_2d - cam->project(T_w_c.inverse() * p_3d_w);

    return true;
  }

  Eigen::Vector2d p_2d;
  std::string cam_model;
};

 *
 *
 *
 *
 */


