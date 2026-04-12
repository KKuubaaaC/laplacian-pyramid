#ifndef SRC_PYRAMID_OPS_H_
#define SRC_PYRAMID_OPS_H_

// INTERNAL HEADER — not part of the public API. Do not include from user code

#include <opencv2/core/mat.hpp>

namespace pyramid {
namespace internal {

// Gaussian blur → downsample: one step down the Gaussian pyramid
[[nodiscard]] cv::Mat Reduce(const cv::Mat& img);

// Upsample → Gaussian blur → ×4: one step up the pyramid
// The ×4 factor compensates for the energy loss introduced by upsampling,
// which fills 3/4 of pixels with zeros (reducing the blurred average by ×4)
[[nodiscard]] cv::Mat Expand(const cv::Mat& img, int target_rows, int target_cols);

}  // namespace internal
}  // namespace pyramid

#endif  // SRC_PYRAMID_OPS_H_
