#pragma once

#include <visnav/common_types.h>
#include <visnav/calibration.h>
#include <pangolin/image/managed_image.h>
#include <ceres/cubic_interpolation.h>
#include <visnav/photometric.h>

namespace visnav {


/// Flags for different landmark outlier criteria
enum PbaOutlierFlags {
  PbaOutlierNone = 0,
  // reprojection error much too large
  PbaOutlierPhotometricErrorHuge = 1 << 0,
  // reprojection error too large
  PbaOutlierPhotometricErrorNormal = 1 << 1,
  // distance to a camera too small
  PbaOutlierCameraDistance = 1 << 2,
  // z-coord in some camera frame too small
  PbaOutlierZCoordinate = 1 << 3
};

/// info on a single projected landmark for pba
struct PbaProjectedLandmark {
  Eigen::Vector2d point_reprojected;            //!< landmark projected into image
  Eigen::Vector3d point_3d_c;                   //!< 3d point in camera coordinates
  TrackId track_id = -1;                        //!< corresponding track_id
  double photometric_error = 0;                 //!< current photometric error
  double depth;                             //!< used for visualization
  unsigned int outlier_flags = PbaOutlierNone;  //!< flags for outlier
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


using PbaProjectedLandmarkPtr = std::shared_ptr<PbaProjectedLandmark>;
using PbaProjectedLandmarkConstPtr = std::shared_ptr<const PbaProjectedLandmark>;


/// landmarks in the map used for photometric ba
struct PbaLandmark {
    /// reference frame of the landmark
    FrameCamId ref_frame;
    /// 2d coordinates of the landmark in the reference frame
    Eigen::Vector2d p_2d;
    /// inverse depth of the landmark wrt the reference frame
    double inv_depth;
    /// intensity of the point
    double intensity;

    /// Inlier observations in the current map.
    /// This is a subset of the original feature track.
    std::vector<FrameCamId> obs; //TODO std::unordered_map is more efficient see sfm.cpp render pba landmarks incam1/2
    //FeatureTrack obs; //TODO another type from FeatureTrack? no need for feature id? when T_w_cs are updated,
                      // we may not observe the landmark in the same frames! how to handle that?


    PbaLandmark() {}

    PbaLandmark(FrameCamId ref_frame, Eigen::Vector2d p_2d, double inv_depth, double intensity,
                std::vector<FrameCamId> obs) : ref_frame(ref_frame), p_2d(p_2d), inv_depth(inv_depth),
                                               intensity(intensity), obs(obs) {}

    /// Construct from landmark
    PbaLandmark(const Landmark& lm,
                const Cameras& cameras,
                const Calibration& calib_cam,
                const tbb::concurrent_unordered_map<FrameCamId, pangolin::ManagedImage<uint8_t>>& images) {
        // Treat the frame of the first observation as the ref frame
        this->ref_frame = lm.obs.begin()->first;

        // Compute inverse depth
        const auto& T_w_c = cameras.at(ref_frame).T_w_c;
        const auto& p_in_ref_frame = T_w_c.inverse() * lm.p;
        this->inv_depth = 1.0 / p_in_ref_frame.norm();

        // Set 2d coordinates
        this->p_2d = calib_cam.intrinsics[ref_frame.cam_id]->project(p_in_ref_frame);

        // Interpolate intensity
        const auto& img = images.at(ref_frame);
        ceres::Grid2D<uint8_t,1> grid(img.begin(), 0, img.h, 0, img.w);
        ceres::BiCubicInterpolator<ceres::Grid2D<uint8_t,1>>  interp(grid);
        interp.Evaluate(p_2d(1), p_2d(0), &(this->intensity));

        update_obs(cameras, calib_cam, images, 25.0);
        /*for(const auto& cam_kv : cameras) {
            if(cam_kv.first == this->ref_frame) {
                continue;
            }

            const auto& T_c1_c0 = cam_kv.second.T_w_c.inverse() * T_w_c;
            const auto& reproj = calib_cam.intrinsics[cam_kv.first.cam_id]->project(T_c1_c0 * p_in_ref_frame);
            if(0 <= reproj.x() && reproj.x() < img.w && 0 <= reproj.y() && reproj.y() < img.h) {
                this->obs.push_back(cam_kv.first);
            }
        }*/
        // Copy observations
        /*for(const auto& obs_kv : lm.obs) {
            this->obs.push_back(obs_kv.first);
        }*/
        //this->outlier_obs = lm.outlier_obs;
    }

    void update_obs(const Cameras& cameras,
                    const Calibration& calib_cam,
                    const tbb::concurrent_unordered_map<FrameCamId, pangolin::ManagedImage<uint8_t>>& images,
                    const double& photometric_threshold) {
        this->obs.clear();

        if(inv_depth < 0.02) {
            return;
        }

        const auto& ref_img = images.at(ref_frame);
        const auto& T_w_c = cameras.at(ref_frame).T_w_c;
        const auto& p_in_ref_frame = calib_cam.intrinsics[ref_frame.cam_id]->unproject(p_2d) / inv_depth;

        for(const auto& cam_kv : cameras) {
            if(cam_kv.first == this->ref_frame) {
                continue;
            }
            const auto& target_img = images.at(cam_kv.first);
            const auto& T_c1_c0 = cam_kv.second.T_w_c.inverse() * T_w_c;
            const auto& reproj = calib_cam.intrinsics[cam_kv.first.cam_id]->project(T_c1_c0 * p_in_ref_frame);
            double residual = compute_residual(cameras, calib_cam, cam_kv.first, ref_img, target_img);
            // Dont add occluded points as an observation and also out of bound obs
            if(residual < photometric_threshold) {
                this->obs.push_back(cam_kv.first);
            }
        }
    }

    double compute_residual(const Cameras& cameras,
                            const Calibration& calib_cam,
                            const FrameCamId& target_fcid,
                            const pangolin::ManagedImage<uint8_t>& ref_img,
                            const pangolin::ManagedImage<uint8_t>& target_img) {
        const auto& T_w_c = cameras.at(ref_frame).T_w_c;
        const auto& target_ab = cameras.at(target_fcid).ab;
        const auto& ref_ab = cameras.at(ref_frame).ab;

        double residual = 0.;
        for(const auto& pattern : residual_pattern) {
            Eigen::Vector2d ref_point(p_2d.x() + pattern.first, p_2d.y() + pattern.second);
            Eigen::Vector3d p_in_ref_frame = calib_cam.intrinsics[ref_frame.cam_id]->unproject(ref_point) / inv_depth;
            const auto& T_c1_c0 = cameras.at(target_fcid).T_w_c.inverse() * T_w_c;
            Eigen::Vector2d target_point = calib_cam.intrinsics[target_fcid.cam_id]->project(T_c1_c0 * p_in_ref_frame);

            if(!target_img.InBounds(target_point.x(), target_point.y(), 0.)) {
                residual = std::numeric_limits<double>::max();
                return residual;
            }

            double ref_intensity = ref_img((size_t) std::round(ref_point.x()), (size_t) std::round(ref_point.y()));
            double target_intensity = target_img((size_t) std::round(target_point.x()), (size_t) std::round(target_point.y()));
            residual += abs((target_intensity - target_ab[1]) - exp(target_ab[0] - ref_ab[0]) * (ref_intensity - ref_ab[1]));
        }
        residual /= residual_pattern.size();
        return residual;
    }

    //const auto& cam = calib_cam.intrinsics[ref_frame.cam_id];

    /// Projects the landmark into the given target frame and
    /// also computes the photometric error
    void compute_projection(const FrameCamId& target_frame,
                            const Cameras& cameras,
                            const Calibration& calib_cam,
                            const pangolin::ManagedImage<uint8_t>& ref_img,
                            const pangolin::ManagedImage<uint8_t>& target_img,
                            PbaProjectedLandmarkPtr& lm_proj) const {

        const auto& ref_intrinsics = calib_cam.intrinsics[ref_frame.cam_id];
        const auto& target_intrinsics = calib_cam.intrinsics[target_frame.cam_id];
        const auto& T_w_ref = cameras.at(ref_frame).T_w_c;
        const auto& T_w_target = cameras.at(target_frame).T_w_c;

        const auto unproj_p_2d = ref_intrinsics->unproject(p_2d);
        const auto point_3d_c = T_w_target.inverse() * T_w_ref * (unproj_p_2d / inv_depth);
        lm_proj->depth = point_3d_c.norm();
        const auto point_reprojected = target_intrinsics->project(point_3d_c);

        // Compute photometric error //TODO compute with residual pattern
        lm_proj->photometric_error = intensity - target_img((size_t) point_reprojected.x(), (size_t) point_reprojected.y());

        lm_proj->point_3d_c = point_3d_c;
        lm_proj->point_reprojected = point_reprojected;
    }

    Eigen::Vector3d get_p(const Cameras& cameras,
                          const Calibration& calib_cam) const {

        const auto& ref_intrinsics = calib_cam.intrinsics[ref_frame.cam_id];
        const auto& T_w_ref = cameras.at(ref_frame).T_w_c;
        const auto unproj_p_2d = ref_intrinsics->unproject(p_2d);
        return T_w_ref * (unproj_p_2d / inv_depth);
    }

};


using PbaLandmarks = std::map<TrackId, PbaLandmark>; //ordered because we want to find the largest track id


/// all landmark projections for inlier and outlier observations for a single
/// image

struct PbaImageProjection {
  std::vector<PbaProjectedLandmarkConstPtr> obs;
};

/// projections for all images
using PbaImageProjections = std::map<FrameCamId, PbaImageProjection>;

/// inlier projections indexed per track
using PbaTrackProjections =
    std::unordered_map<TrackId,
                       std::map<FrameCamId, PbaProjectedLandmarkConstPtr>>;

struct CandidatePoint {
    size_t x;
    size_t y;
    FrameCamId fcid;
};

struct RefPoint {
    Eigen::Vector2d p_2d;
    /// used for visualization
    double depth;
};

/// fcid <=> reference points, used for visualization
using PbaRefPoints = std::unordered_map<FrameCamId, std::vector<RefPoint>>;

}
