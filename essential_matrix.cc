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
#include "colmap/base/essential_matrix.h"
#include "colmap/base/pose.h"
#include "colmap/estimators/essential_matrix.h"
#include "colmap/optim/loransac.h"
#include "colmap/util/random.h"

using namespace colmap;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

py::dict essential_matrix_estimation(
        const std::vector<Eigen::Vector2d> points2D1,
        const std::vector<Eigen::Vector2d> points2D2,
        const py::dict camera_dict1,
        const py::dict camera_dict2,
        const double max_error_px,
        const bool do_refinement
) {
    SetPRNGSeed(0);

    // Check that both vectors have the same size.
    assert(points2D1.size() == points2D2.size());

    // Failure output dictionary.
    py::dict failure_dict;
    failure_dict["success"] = false;

    // Create cameras.
    Camera camera1;
    camera1.SetModelIdFromName(camera_dict1["model"].cast<std::string>());
    camera1.SetWidth(camera_dict1["width"].cast<size_t>());
    camera1.SetHeight(camera_dict1["height"].cast<size_t>());
    camera1.SetParams(camera_dict1["params"].cast<std::vector<double>>());

    Camera camera2;
    camera2.SetModelIdFromName(camera_dict2["model"].cast<std::string>());
    camera2.SetWidth(camera_dict2["width"].cast<size_t>());
    camera2.SetHeight(camera_dict2["height"].cast<size_t>());
    camera2.SetParams(camera_dict2["params"].cast<std::vector<double>>());

    // Image to world.
    std::vector<Eigen::Vector2d> world_points2D1;
    for (size_t idx = 0; idx < points2D1.size(); ++idx) {
        world_points2D1.push_back(camera1.ImageToWorld(points2D1[idx]));
    }

    std::vector<Eigen::Vector2d> world_points2D2;
    for (size_t idx = 0; idx < points2D2.size(); ++idx) {
        world_points2D2.push_back(camera2.ImageToWorld(points2D2[idx]));
    }
    
    // Compute world error.
    const double max_error = 0.5 * (
        max_error_px / camera1.MeanFocalLength() + max_error_px / camera2.MeanFocalLength()
    );

    // Essential matrix estimation parameters.
    RANSACOptions ransac_options;
    ransac_options.max_error = max_error;
    ransac_options.min_inlier_ratio = 0.01;
    ransac_options.min_num_trials = 1000;
    ransac_options.max_num_trials = 100000;
    ransac_options.confidence = 0.9999;
    
    LORANSAC<
        EssentialMatrixFivePointEstimator,
        EssentialMatrixFivePointEstimator
    > ransac(ransac_options);

    // Essential matrix estimation.
    const auto report = ransac.Estimate(world_points2D1, world_points2D2);

    if (!report.success) {
        return failure_dict;
    }

    // Recover data from report.
    const Eigen::Matrix3d E = report.model;
    const size_t num_inliers = report.support.num_inliers;
    const auto inlier_mask = report.inlier_mask;

    // Pose from essential matrix.
    std::vector<Eigen::Vector2d> inlier_world_points2D1;
    std::vector<Eigen::Vector2d> inlier_world_points2D2;

    for (size_t idx = 0; idx < inlier_mask.size(); ++idx) {
        if (inlier_mask[idx]) {
            inlier_world_points2D1.push_back(world_points2D1[idx]);
            inlier_world_points2D2.push_back(world_points2D2[idx]);
        }
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d tvec;
    std::vector<Eigen::Vector3d> points3D;
    PoseFromEssentialMatrix(E, inlier_world_points2D1, inlier_world_points2D2, 
                                                        &R, &tvec, &points3D);

    Eigen::Vector4d qvec = RotationMatrixToQuaternion(R);

    std::string refinement_outcome = "NA";

    if(do_refinement) {
        ceres::Solver::Options options;
        const bool refinement_success = RefineRelativePose(
            options, inlier_world_points2D1, inlier_world_points2D2, &qvec, &tvec);
        if(refinement_success){
            refinement_outcome="True";
        } else {
            refinement_outcome="False";
        }
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
    success_dict["refinement_success"] = refinement_outcome;
    success_dict["E"] = E;
    success_dict["qvec"] = qvec;
    success_dict["tvec"] = tvec;
    success_dict["num_inliers"] = num_inliers;
    success_dict["inliers"] = inliers;
    
    return success_dict;
}
