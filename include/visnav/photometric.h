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



/*
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
        Sophus::SE3<T> T_c2_c1 = T_w_c2.inverse() * T_w_c1;

        //Eigen::Matrix<T, 2, 1> p_target = cam2->project(T_w_c2.inverse() * T_w_c1 * (p_unproj / inv_depth[0]));
        // more stable:
        Eigen::Matrix<T, 2, 1> p_target = cam2->project(T_c2_c1.so3() * p_unproj + inv_depth[0] * T_c2_c1.translation());

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
    const std::string cam_model;
    const ceres::BiCubicInterpolator<ceres::Grid2D<double,1>>& interp_target;
    const size_t h;
    const size_t w;
};
*/

struct BrightnessTransferRegularizer {

    BrightnessTransferRegularizer() {
    }

    template <class T>
    bool operator()(T const* const ab, T* residuals) const {
        residuals[0] = ab[0];
        residuals[1] = T(20.) * ab[1];
        return true;
    }
};


struct PhotometricCostFunctor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PhotometricCostFunctor(const std::vector<Eigen::Vector2d> points,
                           const std::vector<double>& intensities,
                           const std::string cam_model,
                           const ceres::BiCubicInterpolator<ceres::Grid2D<double,1>>& interp_target,
                           const size_t h, const size_t w)
        : points(points), intensities(intensities), cam_model(cam_model), interp_target(interp_target),
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
        Sophus::SE3<T> T_c2_c1 = T_w_c2.inverse() * T_w_c1;

        //Eigen::Matrix<T, 2, 1> p_target = cam2->project(T_w_c2.inverse() * T_w_c1 * (p_unproj / inv_depth[0]));
        // more stable:
        Eigen::Matrix<T, 2, 1> p_target = cam2->project(T_c2_c1.so3() * p_unproj + inv_depth[0] * T_c2_c1.translation());

        return p_target;
    }


    template <class T>
    bool operator()(T const* const sT_w_c1, T const* const sT_w_c2,
                    T const* const sIntr1, T const* const sIntr2,
                    T const* const ab_ref, T const* const ab_target,
                    T const* const inv_depth,
                    T* sResiduals) const {


        Eigen::Map<Eigen::Matrix<T, pattern_size, 1>> residuals(sResiduals);

        if(inv_depth[0] < T(0.)) {
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

        std::vector<Eigen::Matrix<T, 2, 1>> target_points;
        for(size_t i=0; i<points.size(); i++) {
            Eigen::Matrix<T, 2, 1> p = points[i].cast<T>();
            Eigen::Matrix<T, 2, 1> p_target = project_to_target(p, cam1, cam2, T_w_c1, T_w_c2, inv_depth);

            if(p_target(0) < T(0.) || p_target(0) > T(w) || p_target(1) < T(0.) || p_target(1) > T(h)) {
                residuals = Eigen::Matrix<T, pattern_size, 1>::Zero();
                return true;
            }

            target_points.push_back(p_target);
        }

        for(size_t i=0; i<target_points.size(); i++) {
            const auto& p_target = target_points[i];
            T intensity_target;
            interp_target.Evaluate(p_target(1), p_target(0), &intensity_target);

            residuals[i] = (intensity_target - ab_target[1]) - exp(ab_target[0] - ab_ref[0]) * (T(intensities[i]) - ab_ref[1]);
        }

        return true;
    }


    template <class T>
    bool operator()(T const* const sT_w_c1, T const* const sT_w_c2,
                    T const* const sIntr,
                    T const* const ab_ref, T const* const ab_target,
                    T const* const inv_depth,
                    T* sResiduals) const {

        Eigen::Map<Eigen::Matrix<T, pattern_size, 1>> residuals(sResiduals);

        if(inv_depth[0] < T(0.)) {
            residuals = Eigen::Matrix<T, pattern_size, 1>::Zero();
            return true;
        }

        // map inputs
        Eigen::Map<Sophus::SE3<T> const> const T_w_c1(sT_w_c1);
        Eigen::Map<Sophus::SE3<T> const> const T_w_c2(sT_w_c2);

        const std::shared_ptr<AbstractCamera<T>> cam =
            AbstractCamera<T>::from_data(cam_model, sIntr);

        std::vector<Eigen::Matrix<T, 2, 1>> target_points;
        for(size_t i=0; i<points.size(); i++) {
            Eigen::Matrix<T, 2, 1> p = points[i].cast<T>();
            Eigen::Matrix<T, 2, 1> p_target = project_to_target(p, cam, cam, T_w_c1, T_w_c2, inv_depth);

            if(p_target(0) < T(0.) || p_target(0) > T(w) || p_target(1) < T(0.) || p_target(1) > T(h)) {
                residuals = Eigen::Matrix<T, pattern_size, 1>::Zero();
                return true;
            }

            target_points.push_back(p_target);
        }

        for(size_t i=0; i<target_points.size(); i++) {
            const auto& p_target = target_points[i];

            T intensity_target;
            interp_target.Evaluate(p_target(1), p_target(0), &intensity_target);

            residuals[i] = (intensity_target - ab_target[1]) - exp(ab_target[0] - ab_ref[0]) * (T(intensities[i]) - ab_ref[1]);
        }

        return true;
    }

    const std::vector<Eigen::Vector2d> points;
    //const double intensity_ref;
    const std::vector<double>& intensities;
    const std::string cam_model;
    const ceres::BiCubicInterpolator<ceres::Grid2D<double,1>>& interp_target;
    const size_t h;
    const size_t w;
};

}
