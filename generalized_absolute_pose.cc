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
//#include "colmap/estimators/generalized_pose.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/random.h"
#include "colmap/util/misc.h"

#include "colmap/base/camera_models.h"
#include "colmap/optim/bundle_adjustment.h"
#include "colmap/base/cost_functions.h"

using namespace colmap;
using namespace ceres;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/local_parameterization.h>
#include <ceres/internal/port.h>

namespace py = pybind11;

namespace {

typedef RANSAC<GP3PEstimator> GeneralizedAbsolutePoseRANSAC;

}  // namespace

struct GeneralizedAbsolutePoseEstimationOptions {
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

struct GeneralizedAbsolutePoseRefinementOptions {
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

  // Whether to refine the camera rig rotation.
  bool refine_rig_q = false;
  
  // Whether to refine the camera rig translation.
  bool refine_rig_t = false;

  // Whether to refine the camera rig translation.
  bool refine_rig_scale = false;

  // Weight factor for the camera rig cost function
  int rig_cost_weight = 1;

  // Whether to print final summary.
  bool print_summary = true;

  void Check() const {
    CHECK_GE(gradient_tolerance, 0.0);
    CHECK_GE(max_num_iterations, 0);
    CHECK_GE(loss_function_scale, 0.0);
  }
};

//class ScaleParameterization : public LocalParameterization {
class CERES_EXPORT ScaleParameterization : public LocalParameterization {
 public:
  virtual ~ScaleParameterization() {}
  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const {
    const double scale = std::sqrt(std::pow(x[0], 2) + std::pow(x[1], 2) + std::pow(x[2], 2));
    const double new_scale = scale + delta[0];
    x_plus_delta[0] = x[0] * new_scale / scale;
    x_plus_delta[1] = x[1] * new_scale / scale;
    x_plus_delta[2] = x[2] * new_scale / scale  ;
  }
  virtual bool ComputeJacobian(const double* x,
                               double* jacobian) const {
    // J = D_2 [+](x,0)
    // [+](x,d) = x * (s + d) / s = x + x*d/s
    // D_2[+](x,d) = x/a
    // x = x_unit * s
    // Jacobian size: global_size * local_size
    const double scale = std::sqrt(std::pow(x[0], 2) + std::pow(x[1], 2) + std::pow(x[2], 2));
    jacobian[0] = x[0] / scale;
    jacobian[1] = x[1] / scale;
    jacobian[2] = x[2] / scale;
  }
  virtual int GlobalSize() const { return 3; }  // How many total variables in vector
  virtual int LocalSize() const { return 1; }   // How many free parameters
};

class CameraRigRotationCostFunction {
 public:
  CameraRigRotationCostFunction(const Eigen::Vector4d& q, const int& w)
      : q0_(q(0)), q1_(q(1)), q2_(q(2)), q3_(q(3)), w_(w) {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& q, const int& w) {
    return (new ceres::AutoDiffCostFunction<CameraRigRotationCostFunction, 1, 4>(
        new CameraRigRotationCostFunction(q, w)));
  }

  template <typename T>
  bool operator()(const T* const qvec, T* residuals) const {

    T dq = q0_*qvec[0] + q1_*qvec[1] + q2_*qvec[2] + q3_*qvec[3];

    // https://math.stackexchange.com/questions/90081/quaternion-distance
    residuals[0] = T(w_)*(T(1)-pow(dq, 2));

    return true;
  }

 private:
  const double q0_;
  const double q1_;
  const double q2_;
  const double q3_;
  const int w_;
};

class CameraRigScaleCostFunction {
 public:
  CameraRigScaleCostFunction(const Eigen::Vector3d& t, const int& w)
      : t0_(t(0)), t1_(t(1)), t2_(t(2)), w_(w) {}

  static ceres::CostFunction* Create(const Eigen::Vector3d& t, const int& w) {
    return (new ceres::AutoDiffCostFunction<CameraRigScaleCostFunction, 1, 3>(
        new CameraRigScaleCostFunction(t, w)));
  }

  template <typename T>
  bool operator()(const T* const tvec, T* residuals) const {

    T new_scale = ceres::sqrt(pow(tvec[0], 2) + pow(tvec[1], 2) + pow(tvec[2], 2));

    // error = (s_0 - s)^2, squared to make it symmetric and convex
    residuals[0] = T(w_)*pow(orig_scale_ - new_scale, 2);

    return true;
  }

 private:
  const double t0_;
  const double t1_;
  const double t2_;
  const double orig_scale_ = std::sqrt(std::pow(t0_, 2) + std::pow(t1_, 2) + std::pow(t2_, 2));
  const int w_;
};

bool RefineGeneralizedAbsolutePose(
                        const GeneralizedAbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        const std::vector<size_t>& camera_idxs,
                        const std::vector<Eigen::Matrix3x4d>& rel_tforms,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        std::vector<Eigen::Vector4d>* rig_qvecs,
                        std::vector<Eigen::Vector3d>* rig_tvecs,
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

  std::vector<Eigen::Vector4d> rig_qvecs_original;
  std::vector<Eigen::Vector3d> rig_tvecs_original;

  for (size_t i = 0; i < rel_tforms.size(); ++i) {
    Eigen::Vector3d cam_tvec = rel_tforms[i].col(3);
    Eigen::Matrix3d r_mat = rel_tforms[i].leftCols(3);
    Eigen::Vector4d cam_qvec = RotationMatrixToQuaternion(r_mat);
    rig_qvecs_copy.push_back(cam_qvec);
    rig_tvecs_copy.push_back(cam_tvec);
    rig_qvecs_original.push_back(cam_qvec);
    rig_tvecs_original.push_back(cam_tvec);
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

  // Rig cost
  for (size_t i = 0; i < cameras->size(); i++) {
    if (camera_counts[i] == 0)
      continue;
    
    if (options.refine_rig_q) {
      ceres::CostFunction* q_cost_function = nullptr;
      q_cost_function = CameraRigRotationCostFunction::Create(rig_qvecs_original[i], options.rig_cost_weight);
      problem.AddResidualBlock(q_cost_function, loss_function, rig_qvecs_copy[i].data());
    }
    else {
      problem.SetParameterBlockConstant(rig_qvecs_copy[i].data());
    }

    // Don't refine translation of camera 1
    if (options.refine_rig_t && i != 0) {
      ceres::CostFunction* t_cost_function = nullptr;
      t_cost_function = CameraRigScaleCostFunction::Create(rig_tvecs_original[i], options.rig_cost_weight);
      problem.AddResidualBlock(t_cost_function, loss_function, rig_tvecs_copy[i].data());
    }
    else {
      problem.SetParameterBlockConstant(rig_tvecs_copy[i].data());
    }
  } 

  if (problem.NumResiduals() > 0) {
    // Quaternion parameterization.
    *qvec = NormalizeQuaternion(*qvec);
    ceres::LocalParameterization* quaternion_parameterization =
        new ceres::QuaternionParameterization;
    problem.SetParameterization(qvec_data, quaternion_parameterization);

    ceres::LocalParameterization* scale_parameterization =
        new ScaleParameterization;

    // Camera parameterization.
    for (size_t i = 0; i < cameras->size(); i++) {
      if (camera_counts[i] == 0)
        continue;
      Camera& camera = cameras->at(i);

      if (options.refine_rig_scale && i != 0) {
        problem.SetParameterization(rig_tvecs_copy[i].data(), scale_parameterization);
      }
      problem.SetParameterization(rig_qvecs_copy[i].data(), quaternion_parameterization);

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

  for (size_t i = 0; i < cameras->size(); i++) {
    rig_qvecs->push_back(rig_qvecs_copy[i]);
    rig_tvecs->push_back(rig_tvecs_copy[i]);
  }

  return summary.IsSolutionUsable();
}

bool EstimateGeneralizedAbsolutePose(
                        const GeneralizedAbsolutePoseEstimationOptions& options,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        const std::vector<size_t>& cam_idxs,                                
                        const std::vector<Eigen::Matrix3x4d>& rel_camera_poses,
                        const std::vector<Camera>& cameras,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        size_t* num_inliers,
                        std::vector<char>* inlier_mask) {
  options.Check();

  // TODO(sebgiles): check cameras and tforms are same length

  // Normalize image coordinates.
  std::vector<Eigen::Vector2d> points2D_normalized(points2D.size());
  for (size_t i = 0; i < points2D.size(); ++i) {
    points2D_normalized[i] = cameras[cam_idxs[i]].ImageToWorld(points2D[i]);
  }

  // Format data for the solver.
  std::vector<GP3PEstimator::X_t> points2D_with_tf;
  for (size_t i = 0; i < points2D_normalized.size(); ++i) {
    points2D_with_tf.emplace_back();
    points2D_with_tf.back().rel_tform = rel_camera_poses[cam_idxs[i]];
    points2D_with_tf.back().xy = points2D_normalized[i];
  }

  // Estimate pose for given focal length.
  auto custom_options = options;
  custom_options.ransac_options.max_error = 
      cameras[0].ImageToWorldThreshold(options.ransac_options.max_error);
  GeneralizedAbsolutePoseRANSAC ransac(custom_options.ransac_options);
  auto report = ransac.Estimate(points2D_with_tf, points3D);

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

py::dict generalized_absolute_pose_estimation(
        const std::vector<Eigen::Vector2d> points2D,
        const std::vector<Eigen::Vector3d> points3D,
        const std::vector<size_t> cam_idxs,
        const std::vector<Eigen::Matrix3x4d> rel_camera_poses,
        const std::vector<py::dict> camera_dicts,
        const py::dict refinement_parameters,
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
    abs_pose_refinement_options.refine_rig_t = refinement_parameters["refine_t"].cast<bool>();
    abs_pose_refinement_options.refine_rig_scale = refinement_parameters["refine_scale"].cast<bool>();
    abs_pose_refinement_options.refine_rig_q = refinement_parameters["refine_q"].cast<bool>();
    abs_pose_refinement_options.rig_cost_weight = refinement_parameters["rig_cost_weight"].cast<int>();


    std::vector<Eigen::Vector4d> rig_qvecs;
    std::vector<Eigen::Vector3d> rig_tvecs;

    // Absolute pose refinement.
    if (!RefineGeneralizedAbsolutePose(abs_pose_refinement_options, inlier_mask, points2D, 
            points3D, cam_idxs, rel_camera_poses, &qvec, &tvec, &rig_qvecs, &rig_tvecs, &cameras)) {
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
    success_dict["rig_qvecs"] = rig_qvecs;
    success_dict["rig_tvecs"] = rig_tvecs;
    success_dict["num_inliers"] = num_inliers;
    success_dict["inliers"] = inliers;
    
    return success_dict;
}
