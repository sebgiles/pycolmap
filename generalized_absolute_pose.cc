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
#include "colmap/base/pose.h"
#include "colmap/base/projection.h"
#include "colmap/base/similarity_transform.h"
#include "colmap/estimators/generalized_absolute_pose.h"
#include "colmap/estimators/generalized_pose.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/random.h"
#include "colmap/util/misc.h"

#include "colmap/base/camera_models.h"
#include "colmap/optim/bundle_adjustment.h"
#include "colmap/base/cost_functions.h"

using namespace colmap;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

bool RefineGeneralizedAbsolutePose(
                        const GeneralizedAbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        const std::vector<size_t>& camera_idxs,
                        const std::vector<Eigen::Matrix3x4d>& rel_tforms,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        std::vector<Camera>* cameras) {
  CHECK_EQ(points2D.size(), inlier_mask.size());
  CHECK_EQ(points2D.size(), points3D.size());
  CHECK_EQ(points2D.size(), camera_idxs.size());
  //CHECK_EQ(rig_qvecs.size(), rig_tvecs.size());
  //CHECK_EQ(rig_qvecs.size(), cameras->size());
  CHECK_GE(*std::min_element(camera_idxs.begin(), camera_idxs.end()), 0);
  CHECK_LT(*std::max_element(camera_idxs.begin(), camera_idxs.end()), cameras->size());
  options.Check();

  ceres::LossFunction* loss_function =
      new ceres::CauchyLoss(options.loss_function_scale);

  std::vector<double*> cameras_params_data;
  for (size_t i = 0; i < cameras->size(); i++) {
    cameras_params_data.push_back(cameras->at(i).ParamsData());
  }
  std::vector<size_t> camera_counts(cameras->size(), 0);
  double* qvec_data = qvec->data();
  double* tvec_data = tvec->data();

  std::vector<Eigen::Vector3d> points3D_copy = points3D;
  std::vector<Eigen::Vector4d> rig_qvecs_copy;
  std::vector<Eigen::Vector3d> rig_tvecs_copy;

  for (size_t i = 0; i < rel_tforms.size(); ++i) {
    Eigen::Vector3d cam_tvec = rel_tforms[i].col(3);
    Eigen::Matrix3d r_mat = rel_tforms[i].leftCols(3);
    Eigen::Vector4d cam_qvec = RotationMatrixToQuaternion(r_mat);
    rig_qvecs_copy.push_back(cam_qvec);
    rig_tvecs_copy.push_back(cam_tvec);
  }

  ceres::Problem problem;

  for (size_t i = 0; i < points2D.size(); ++i) {
    // Skip outlier observations
    if (!inlier_mask[i]) {
      continue;
    }
    size_t camera_idx = camera_idxs[i];
    camera_counts[camera_idx] += 1;

    ceres::CostFunction* cost_function = nullptr;
    switch (cameras->at(camera_idx).ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                  \
  case CameraModel::kModelId:                                           \
    cost_function =                                                     \
        RigBundleAdjustmentCostFunction<CameraModel>::Create(points2D[i]); \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    problem.AddResidualBlock(cost_function, loss_function,
                             qvec_data, tvec_data,
                             rig_qvecs_copy[camera_idx].data(),
                             rig_tvecs_copy[camera_idx].data(),
                             points3D_copy[i].data(),
                             cameras_params_data[camera_idx]);
    problem.SetParameterBlockConstant(points3D_copy[i].data());
  }

  if (problem.NumResiduals() > 0) {
    // Quaternion parameterization.
    *qvec = NormalizeQuaternion(*qvec);
    ceres::LocalParameterization* quaternion_parameterization =
        new ceres::QuaternionParameterization;
    problem.SetParameterization(qvec_data, quaternion_parameterization);

    // Camera parameterization.
    for (size_t i = 0; i < cameras->size(); i++) {
      if (camera_counts[i] == 0)
        continue;
      Camera& camera = cameras->at(i);

      // We don't optimize the rig parameters (it's likely very unconstrainted)
      problem.SetParameterBlockConstant(rig_qvecs_copy[i].data());
      problem.SetParameterBlockConstant(rig_tvecs_copy[i].data());

      if (!options.refine_focal_length && !options.refine_extra_params) {
        problem.SetParameterBlockConstant(camera.ParamsData());
      } else {
        // Always set the principal point as fixed.
        std::vector<int> camera_params_const;
        const std::vector<size_t>& principal_point_idxs =
            camera.PrincipalPointIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   principal_point_idxs.begin(),
                                   principal_point_idxs.end());

        if (!options.refine_focal_length) {
          const std::vector<size_t>& focal_length_idxs =
              camera.FocalLengthIdxs();
          camera_params_const.insert(camera_params_const.end(),
                                     focal_length_idxs.begin(),
                                     focal_length_idxs.end());
        }

        if (!options.refine_extra_params) {
          const std::vector<size_t>& extra_params_idxs =
              camera.ExtraParamsIdxs();
          camera_params_const.insert(camera_params_const.end(),
                                     extra_params_idxs.begin(),
                                     extra_params_idxs.end());
        }

        if (camera_params_const.size() == camera.NumParams()) {
          problem.SetParameterBlockConstant(camera.ParamsData());
        } else {
          ceres::SubsetParameterization* camera_params_parameterization =
              new ceres::SubsetParameterization(
                  static_cast<int>(camera.NumParams()), camera_params_const);
          problem.SetParameterization(camera.ParamsData(),
                                      camera_params_parameterization);
        }
      }
    }
  }

  ceres::Solver::Options solver_options;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  //solver_options.minimizer_progress_to_stdout = true;  // fixme

  // The overhead of creating threads is too large.
  solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options.print_summary) {
    PrintHeading2("Pose refinement report");
    PrintSolverSummary(summary);
  }

  return summary.IsSolutionUsable();
}

py::dict generalized_absolute_pose_estimation(
        const std::vector<Eigen::Vector2d> points2D,
        const std::vector<Eigen::Vector3d> points3D,
        const std::vector<size_t> cam_idxs,
        const std::vector<Eigen::Matrix3x4d> rel_camera_poses,
        const std::vector<py::dict> camera_dicts,
        const double max_error_px
) {
    SetPRNGSeed(0);

    // Check that both vectors have the same size.
    assert(points2D.size() == points3D.size());

    // Failure output dictionary.
    py::dict failure_dict;
    failure_dict["success"] = false;

    // Create cameras.
    std::vector<Camera> cameras;
    for (auto& camera_dict: camera_dicts) {
        cameras.emplace_back();
        cameras.back().SetModelIdFromName(
                camera_dict["model"].cast<std::string>());
        cameras.back().SetWidth(camera_dict["width"].cast<size_t>());
        cameras.back().SetHeight(camera_dict["height"].cast<size_t>());
        cameras.back().SetParams(
                camera_dict["params"].cast<std::vector<double>>());
    }

    // Absolute pose estimation parameters.
    GeneralizedAbsolutePoseEstimationOptions abs_pose_options;
    abs_pose_options.estimate_focal_length = false;
    abs_pose_options.ransac_options.max_error = max_error_px;
    abs_pose_options.ransac_options.min_inlier_ratio = 0.01;
    abs_pose_options.ransac_options.min_num_trials = 1000;
    abs_pose_options.ransac_options.max_num_trials = 100000;
    abs_pose_options.ransac_options.confidence = 0.9999;

    // Absolute pose estimation.
    Eigen::Vector4d qvec;
    Eigen::Vector3d tvec;
    size_t num_inliers;
    std::vector<char> inlier_mask;

    if (!EstimateGeneralizedAbsolutePose(abs_pose_options, 
            points2D, points3D, cam_idxs, rel_camera_poses, cameras, 
            &qvec, &tvec, &num_inliers, &inlier_mask)) {
        return failure_dict;
    }

    // Refine absolute pose parameters.
    GeneralizedAbsolutePoseRefinementOptions abs_pose_refinement_options;
    abs_pose_refinement_options.refine_focal_length = false;
    abs_pose_refinement_options.refine_extra_params = false;
    abs_pose_refinement_options.print_summary = false;

    // Absolute pose refinement.
    if (!RefineGeneralizedAbsolutePose(abs_pose_refinement_options, inlier_mask, points2D, 
            points3D, cam_idxs, rel_camera_poses, &qvec, &tvec, &cameras)) {
        return failure_dict;
    }

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
