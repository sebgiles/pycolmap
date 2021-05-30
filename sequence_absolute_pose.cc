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
#include "colmap/estimators/pose.h"
#include "colmap/util/random.h"

#include "colmap/base/camera_models.h"
#include "colmap/base/cost_functions.h"
#include "colmap/base/essential_matrix.h"
#include "colmap/base/pose.h"
#include "colmap/estimators/absolute_pose.h"
#include "colmap/estimators/essential_matrix.h"
#include "colmap/optim/bundle_adjustment.h"
#include "colmap/util/matrix.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

using namespace colmap;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

bool RefineSequencePose(const std::vector<char>& inlier_mask_0,
                        const std::vector<char>& inlier_mask_1,
                        const std::vector<Eigen::Vector3d> points3D_0,
                        const std::vector<Eigen::Vector3d> points3D_1,
                        const std::vector<Eigen::Vector2d> map_points2D_0,
                        const std::vector<Eigen::Vector2d> map_points2D_1,
                        const std::vector<Eigen::Vector2d> rel1_points2D_0,
                        const std::vector<Eigen::Vector2d> rel0_points2D_1,
                        Eigen::Vector4d* qvec_0, Eigen::Vector3d* tvec_0,
                        Eigen::Vector4d* qvec_1, Eigen::Vector3d* tvec_1,
                        Camera* camera) {

  CHECK_EQ(     points3D_0.size(), inlier_mask_0.size());
  CHECK_EQ(     points3D_1.size(), inlier_mask_1.size());
  CHECK_EQ(     points3D_0.size(), map_points2D_0.size());
  CHECK_EQ(     points3D_1.size(), map_points2D_1.size());
  CHECK_EQ(rel1_points2D_0.size(), rel0_points2D_1.size());

  // Refine absolute pose parameters.
  AbsolutePoseRefinementOptions options;
  options.refine_focal_length = false;
  options.refine_extra_params = false;
  options.print_summary = false;


  ceres::LossFunction* loss_function =
      new ceres::CauchyLoss(options.loss_function_scale);

  double* camera_params_data = camera->ParamsData();
  double* qvec_0_data = qvec_0->data();
  double* tvec_0_data = tvec_0->data();

  std::vector<Eigen::Vector3d> points3D_0_copy = points3D_0;

  ceres::Problem problem;

  for (size_t i = 0; i < map_points2D_0.size(); ++i) {
    // Skip outlier observations
    if (!inlier_mask_0[i]) {
      continue;
    }

    ceres::CostFunction* cost_function = nullptr;

    switch (camera->ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                  \
  case CameraModel::kModelId:                                           \
    cost_function =                                                     \
        BundleAdjustmentCostFunction<CameraModel>::Create(map_points2D_0[i]); \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    problem.AddResidualBlock( cost_function, loss_function, 
                              qvec_0_data, tvec_0_data,
                              points3D_0_copy[i].data(), 
                              camera_params_data);
    problem.SetParameterBlockConstant(points3D_0_copy[i].data());
  }

  if (problem.NumResiduals() > 0) {
    // Quaternion parameterization.
    *qvec_0 = NormalizeQuaternion(*qvec_0);
    ceres::LocalParameterization* quaternion_parameterization =
        new ceres::QuaternionParameterization;
    problem.SetParameterization(qvec_0_data, quaternion_parameterization);

    // Camera parameterization.
    problem.SetParameterBlockConstant(camera->ParamsData());
    
  }

  ceres::Solver::Options solver_options;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::DENSE_QR;

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











py::dict sequence_pose_estimation(
                            const std::vector<Eigen::Vector3d> points3D_0,
                            const std::vector<Eigen::Vector3d> points3D_1,
                            const std::vector<Eigen::Vector2d> map_points2D_0,
                            const std::vector<Eigen::Vector2d> map_points2D_1,
                            const std::vector<Eigen::Vector2d> rel1_points2D_0,
                            const std::vector<Eigen::Vector2d> rel0_points2D_1,
                            const py::dict camera_dict,
                            const double max_error_px) {
    SetPRNGSeed(0);

  // Check that both vectors have the same size.
  assert(     points3D_0.size() ==  map_points2D_0.size());
  assert(     points3D_1.size() ==  map_points2D_1.size());
  assert(rel1_points2D_0.size() == rel0_points2D_1.size());

  // Failure output dictionary.
  py::dict failure_dict;
  failure_dict["success"] = false;

  // Create camera.
  Camera camera;
  camera.SetModelIdFromName(camera_dict["model"].cast<std::string>());
  camera.SetWidth(camera_dict["width"].cast<size_t>());
  camera.SetHeight(camera_dict["height"].cast<size_t>());
  camera.SetParams(camera_dict["params"].cast<std::vector<double>>());

  // Absolute pose estimation parameters.
  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.estimate_focal_length = false;
  abs_pose_options.ransac_options.max_error = max_error_px;
  abs_pose_options.ransac_options.min_inlier_ratio = 0.01;
  abs_pose_options.ransac_options.min_num_trials = 1000;
  abs_pose_options.ransac_options.max_num_trials = 100000;
  abs_pose_options.ransac_options.confidence = 0.9999;

  // Absolute pose estimation. ----------------------------------------------
  Eigen::Vector4d qvec_0;
  Eigen::Vector3d tvec_0;
  size_t num_inliers_0;
  std::vector<char> inlier_mask_0;

  if (!EstimateAbsolutePose(abs_pose_options, 
                                  map_points2D_0, points3D_0, 
                                  &qvec_0, &tvec_0, 
                                  &camera, &num_inliers_0, &inlier_mask_0)) {
    return failure_dict;
  }
  
  Eigen::Vector4d qvec_1;
  Eigen::Vector3d tvec_1;
  size_t num_inliers_1;
  std::vector<char> inlier_mask_1;

  if (!EstimateAbsolutePose(abs_pose_options, 
                                  map_points2D_1, points3D_1, 
                                  &qvec_1, &tvec_1, 
                                  &camera, &num_inliers_1, &inlier_mask_1)) {
    return failure_dict;
  }


  // refinement --------------------------------------------------------------
  if (!RefineSequencePose(inlier_mask_0,      inlier_mask_1, 
                          points3D_0,         points3D_1,
                          map_points2D_0,     map_points2D_1,
                          rel1_points2D_0,    rel0_points2D_1,
                          &qvec_0,            &tvec_0, 
                          &qvec_1,            &tvec_1, 
                          &camera)) {
    return failure_dict;
  }

  // Convert vector<char> to vector<int>.
  std::vector<bool> inliers_0;
  for (auto it : inlier_mask_0) {
    if (it) {
      inliers_0.push_back(true);
    } else {
      inliers_0.push_back(false);
    }
  }

  // Convert vector<char> to vector<int>.
  std::vector<bool> inliers_1;
  for (auto it : inlier_mask_1) {
    if (it) {
      inliers_1.push_back(true);
    } else {
      inliers_1.push_back(false);
    }
  }

  // Success output dictionary.
  py::dict success_dict;
  success_dict["success"] = true;
  success_dict["qvec_0"] = qvec_0;
  success_dict["tvec_0"] = tvec_0;
  success_dict["qvec_0"] = qvec_1;
  success_dict["tvec_0"] = tvec_1;
  success_dict["num_inliers_0"] = num_inliers_0;
  success_dict["inliers_0"] = inliers_0;
  success_dict["num_inliers_1"] = num_inliers_1;
  success_dict["inliers_1"] = inliers_1;
  
  return success_dict;
}
