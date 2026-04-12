#ifndef PYRAMID_PYRAMID_BLEND_H_
#define PYRAMID_PYRAMID_BLEND_H_

#include <opencv2/core/mat.hpp>
#include <optional>

#include "pyramid/pyramid_types.h"

namespace pyramid {

// Multiresolution spline blend (Burt & Adelson): at each pyramid level i,
// LS[i] = GM[i] * LA[i] + (1 - GM[i]) * LB[i], then collapse to a single image
//
// image_a, image_b: CV_32FC1, same size, typically in [0, 1]
// mask: CV_32FC1, same size; 1 selects A, 0 selects B
[[nodiscard]] std::optional<cv::Mat> BlendLaplacianPyramids(const cv::Mat& image_a,
                                                            const cv::Mat& image_b,
                                                            const cv::Mat& mask,
                                                            PyramidParams params = {});

}  // namespace pyramid

#endif  // PYRAMID_PYRAMID_BLEND_H_
