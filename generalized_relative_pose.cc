// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

#include <iostream>
#include <fstream>

#include "colmap/base/camera.h"
#include "colmap/estimators/generalized_relative_pose.h"
#include "colmap/util/random.h"

using namespace colmap;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

namespace {

typedef LORANSAC<GR6PEstimator, GR6PEstimator> GeneralizedRelativePoseRANSAC;

}  // namespace

struct GeneralizedRelativePoseEstimationOptions {
  // Whether to estimate the focal length.
  bool estimate_focal_length = false;

  // Number of discrete samples for focal length estimation.
  size_t num_focal_length_samples = 30;

  // Minimum focal length ratio for discrete focal length sampling
  // around focal length of given camera.
  double min_focal_length_ratio = 0.2;

  // Maximum focal length ratio for discrete focal length sampling
  // around focal length of given camera.
  double max_focal_length_ratio = 5;

  // Number of threads for parallel estimation of focal length.
  int num_threads = ThreadPool::kMaxNumThreads;

  // Options used for P3P RANSAC.
  RANSACOptions ransac_options;

  void Check() const {
    CHECK_GT(num_focal_length_samples, 0);
    CHECK_GT(min_focal_length_ratio, 0);
    CHECK_GT(max_focal_length_ratio, 0);
    CHECK_LT(min_focal_length_ratio, max_focal_length_ratio);
    ransac_options.Check();
  }
};

struct GeneralizedRelativePoseRefinementOptions {
  // Convergence criterion.
  double gradient_tolerance = 1.0;

  // Maximum number of solver iterations.
  int max_num_iterations = 100;

  // Scaling factor determines at which residual robustification takes place.
  double loss_function_scale = 1.0;

  // Whether to refine the focal length parameter group.
  bool refine_focal_length = true;

  // Whether to refine the extra parameter group.
  bool refine_extra_params = true;

  // Whether to print final summary.
  bool print_summary = true;

  void Check() const {
    CHECK_GE(gradient_tolerance, 0.0);
    CHECK_GE(max_num_iterations, 0);
    CHECK_GE(loss_function_scale, 0.0);
  }
};

bool EstimateGeneralizedRelativePose(
                        const GeneralizedRelativePoseEstimationOptions& options,
                        const std::vector<Eigen::Vector2d>& points0,
                        const std::vector<Eigen::Vector2d>& points1,
                        const std::vector<size_t>& cam_idxs0,                                
                        const std::vector<size_t>& cam_idxs1,                                
                        const std::vector<Eigen::Matrix3x4d>& rel_camera_poses,
                        const std::vector<Camera>& cameras,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        size_t* num_inliers,
                        std::vector<char>* inlier_mask) {
  options.Check();

  // TODO(sebgiles): check cameras and tforms are same length

  // Normalize image coordinates.
  std::vector<Eigen::Vector2d> points0_normalized(points0.size());
  for (size_t i = 0; i < points0.size(); ++i) {
    points0_normalized[i] = cameras[cam_idxs0[i]].ImageToWorld(points0[i]);
  }
  std::vector<Eigen::Vector2d> points1_normalized(points1.size());
  for (size_t i = 0; i < points1.size(); ++i) {
    points1_normalized[i] = cameras[cam_idxs1[i]].ImageToWorld(points1[i]);
  }

  // Format data for the solver.
  std::vector<GR6PEstimator::X_t> points0_with_tf;
  for (size_t i = 0; i < points0_normalized.size(); ++i) {
    points0_with_tf.emplace_back();
    points0_with_tf.back().rel_tform = rel_camera_poses[cam_idxs0[i]];
    points0_with_tf.back().xy = points0_normalized[i];
  }
  // Format data for the solver.
  std::vector<GR6PEstimator::X_t> points1_with_tf;
  for (size_t i = 0; i < points1_normalized.size(); ++i) {
    points1_with_tf.emplace_back();
    points1_with_tf.back().rel_tform = rel_camera_poses[cam_idxs1[i]];
    points1_with_tf.back().xy = points1_normalized[i];
  }

  // Estimate pose for given focal length.
  auto custom_options = options;
  custom_options.ransac_options.max_error = 
      cameras[0].ImageToWorldThreshold(options.ransac_options.max_error);
  GeneralizedRelativePoseRANSAC ransac(custom_options.ransac_options);
  auto report = ransac.Estimate(points0_with_tf, points1_with_tf);

  Eigen::Matrix3x4d proj_matrix;
  inlier_mask->clear();

  *num_inliers = report.support.num_inliers;
  proj_matrix = report.model;
  *inlier_mask = report.inlier_mask;

  if (*num_inliers == 0) {
    return false;
  }

  // Extract pose parameters.
  *qvec = RotationMatrixToQuaternion(proj_matrix.leftCols<3>());
  *tvec = proj_matrix.rightCols<1>();

  if (IsNaN(*qvec) || IsNaN(*tvec)) {
    return false;
  }

  return true;
}

py::dict generalized_relative_pose_estimation(
        const std::vector<Eigen::Vector2d> points0,
        const std::vector<Eigen::Vector2d> points1,
        const std::vector<size_t> cam_idxs0,
        const std::vector<size_t> cam_idxs1,
        const std::vector<Eigen::Matrix3x4d> rel_camera_poses,
        const std::vector<py::dict> camera_dicts,
        const double max_error_px
) {
    SetPRNGSeed(0);

    // Check that both vectors have the same size.
    assert(points0.size() == points1.size());
    assert(camera_dicts.size() == rel_camera_poses.size());

    // Failure output dictionary.
    py::dict failure_dict;
    failure_dict["success"] = false;

    // Create cameras.
    std::vector<Camera> cameras;
    for (auto& camera_dict: camera_dicts) {
        cameras.emplace_back();
        cameras.back().SetModelIdFromName(
                //camera_dict["model"].cast<std::string>());
                "OPENCV");
        cameras.back().SetWidth(camera_dict["width"].cast<size_t>());
        cameras.back().SetHeight(camera_dict["height"].cast<size_t>());
        cameras.back().SetParams(
                camera_dict["params"].cast<std::vector<double>>());
    }

    // Relative pose estimation parameters.
    GeneralizedRelativePoseEstimationOptions abs_pose_options;
    abs_pose_options.estimate_focal_length = false;
    abs_pose_options.ransac_options.max_error = max_error_px;
    abs_pose_options.ransac_options.min_inlier_ratio = 0.01;
    abs_pose_options.ransac_options.min_num_trials = 1000;
    abs_pose_options.ransac_options.max_num_trials = 100000;
    abs_pose_options.ransac_options.confidence = 0.9999;

    // Relative pose estimation.
    Eigen::Vector4d qvec;
    Eigen::Vector3d tvec;
    size_t num_inliers;
    std::vector<char> inlier_mask;

    if (!EstimateGeneralizedRelativePose(abs_pose_options, 
            points0, points1, cam_idxs0, cam_idxs1, rel_camera_poses, cameras, 
            &qvec, &tvec, &num_inliers, &inlier_mask)) {
        return failure_dict;
    }

    // Refine absolute pose parameters.
    // RelativePoseRefinementOptions abs_pose_refinement_options;
    // abs_pose_refinement_options.refine_focal_length = false;
    // abs_pose_refinement_options.refine_extra_params = false;
    // abs_pose_refinement_options.print_summary = false;

    // Relative pose refinement.
    // if (!RefineRelativePose(abs_pose_refinement_options, inlier_mask, points2D, 
    //         points3D, rel_camera_poses, &qvec, &tvec, &camera)) {
    //     return failure_dict;
    // }

    // Convert vector<char> to vector<int>.
    std::vector<bool> inliers;
    for (auto it : inlier_mask) {
        if (it) {
            inliers.push_back(true);
        } else {
            inliers.push_back(false);
        }
    }

    // Success output dictionary.
    py::dict success_dict;
    success_dict["success"] = true;
    success_dict["qvec"] = qvec;
    success_dict["tvec"] = tvec;
    success_dict["num_inliers"] = num_inliers;
    success_dict["inliers"] = inliers;
    
    return success_dict;
}
