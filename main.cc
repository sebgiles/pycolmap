#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "absolute_pose.cc"
#include "sequence_absolute_pose.cc"
#include "generalized_absolute_pose.cc"
#include "essential_matrix.cc"
#include "fundamental_matrix.cc"
#include "transformations.cc"
#include "sift.cc"

PYBIND11_MODULE(pycolmap, m) {
    m.doc() = "COLMAP plugin built on " __TIMESTAMP__;

    // Absolute pose.
    m.def("absolute_pose_estimation", &absolute_pose_estimation,
          py::arg("points2D"), py::arg("points3D"),
          py::arg("camera_dict"),
          py::arg("max_error_px") = 12.0,
          "Absolute pose estimation with non-linear refinement.");

    // Absolute pose from Sequence.
    m.def("sequence_pose_estimation", &sequence_pose_estimation,
            py::arg("points3D_0"),
            py::arg("points3D_1"),
            py::arg("map_points2D_0"),
            py::arg("map_points2D_1"),
            py::arg("rel1_points2D_0"),
            py::arg("rel0_points2D_1"),
            py::arg("camera_dict"),
            py::arg("max_error_px") = 12.0,
            py::arg("rel_max_error_px") = 12.0,
            py::arg("rel_weight") = 1000.0,
            "Absolute pose estimation with non-linear refinement.");

    // Absolute pose from multiple images.
    m.def("generalized_absolute_pose_estimation", &generalized_absolute_pose_estimation,
          py::arg("points2D"), py::arg("points3D"), 
          py::arg("cam_idxs"),
          py::arg("rel_camera_poses"),
          py::arg("camera_dicts"),
          py::arg("refinement_parameters"),
          py::arg("max_error_px") = 12.0,
          "Multi image absolute pose estimation.");

    // Essential matrix.
    m.def("essential_matrix_estimation", &essential_matrix_estimation,
          py::arg("points2D1"), py::arg("points2D2"),
          py::arg("camera_dict1"), py::arg("camera_dict2"),
          py::arg("max_error_px") = 4.0,
          py::arg("do_refinement") = false,
          "LORANSAC + 5-point algorithm.");

    // Fundamental matrix.
    m.def("fundamental_matrix_estimation", &fundamental_matrix_estimation,
          py::arg("points2D1"), py::arg("points2D2"),
          py::arg("max_error_px") = 4.0,
          "LORANSAC + 7-point algorithm.");

    // Image-to-world and world-to-image.
    m.def("image_to_world", &image_to_world, "Image to world transformation.");
    m.def("world_to_image", &world_to_image, "World to image transformation.");

    // SIFT.
    m.def("extract_sift", &extract_sift,
          py::arg("image"),
          py::arg("num_octaves") = 4, py::arg("octave_resolution") = 3, py::arg("first_octave") = 0,
          py::arg("edge_thresh") = 10.0, py::arg("peak_thresh") = 0.01, py::arg("upright") = false,
          "Extract SIFT features.");
}
