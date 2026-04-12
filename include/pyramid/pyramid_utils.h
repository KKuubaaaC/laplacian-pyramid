#ifndef PYRAMID_PYRAMID_UTILS_H_
#define PYRAMID_PYRAMID_UTILS_H_

#include <opencv2/core/mat.hpp>

namespace pyramid {

// Peak SNR in dB for CV_32FC1 images of identical size. Uses max_signal^2 in the
// numerator (default max_signal = 1.0 for images in [0, 1]). Returns +infinity
// when MSE is exactly zero.
[[nodiscard]] double ComputePSNR(const cv::Mat& original, const cv::Mat& reconstructed,
                                 double max_signal = 1.0);

}  // namespace pyramid

#endif  // PYRAMID_PYRAMID_UTILS_H_
